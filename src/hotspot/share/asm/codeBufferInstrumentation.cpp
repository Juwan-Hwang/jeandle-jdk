#include "precompiled.hpp"
#include "asm/codeBufferInstrumentation.hpp"
#include "runtime/globals_shared.hpp"
#include "compiler/compiler_globals.hpp"
#include "utilities/ostream.hpp"
#include "runtime/os.hpp"

CodeBufferInstrumentation* CodeBufferInstrumentation::_instance = nullptr;

CodeBufferInstrumentation* CodeBufferInstrumentation::instance() {
  if (_instance == nullptr) {
    _instance = new CodeBufferInstrumentation();
  }
  return _instance;
}

void CodeBufferInstrumentation::record_initialize(const char* name, int code_size,
                                                   int consts_cap, int stubs_size,
                                                   int locs_size) {
  if (!JeandleCodeBufferInstrument) return;
  Record* r = new Record(name, code_size, consts_cap, stubs_size, locs_size);
  instance()->_records.append(r);
}

void CodeBufferInstrumentation::record_expand(int section, int amount,
                                               int new_total_cap, jlong elapsed_us) {
  if (!JeandleCodeBufferInstrument) return;
  CodeBufferInstrumentation* inst = instance();
  if (inst->_records.is_empty()) return;
  Record* r = inst->_records.at(inst->current_index());
  ExpansionEvent e = {section, amount, new_total_cap, elapsed_us};
  r->_expansions.append(e);
  r->_expand_count++;
  r->_total_expand_bytes += amount;
}

void CodeBufferInstrumentation::record_finalize(int consts_size, int consts_cap,
                                                 int insts_size, int stubs_size,
                                                 jlong finalize_latency_us) {
  if (!JeandleCodeBufferInstrument) return;
  CodeBufferInstrumentation* inst = instance();
  if (inst->_records.is_empty()) return;
  Record* r = inst->_records.at(inst->current_index());
  r->_final_consts_size = consts_size;
  r->_final_consts_capacity = consts_cap;
  r->_final_insts_size = insts_size;
  r->_final_stubs_size = stubs_size;
  r->_finalize_latency_us = finalize_latency_us;
}

void CodeBufferInstrumentation::output_json() {
  if (_output_done) return;
  _output_done = true;

  const char* path = "/tmp/codebuffer_instrument.json";
  FILE* fp = os::fopen(path, "w");
  if (fp == nullptr) {
    tty->print_cr("CodeBufferInstrumentation: failed to open %s", path);
    return;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"instrumentation\": \"codebuffer\",\n");
  fprintf(fp, "  \"strategy\": \"legacy_fixed_size\",\n");
  fprintf(fp, "  \"records\": [\n");

  for (int i = 0; i < _records.length(); i++) {
    Record* r = _records.at(i);
    int slack = r->consts_slack();
    double waste = r->consts_waste_rate();

    fprintf(fp, "    {\n");
    fprintf(fp, "      \"index\": %d,\n", i);
    fprintf(fp, "      \"name\": \"%s\",\n", r->_name ? r->_name : "");
    fprintf(fp, "      \"initial\": {\n");
    fprintf(fp, "        \"code_size\": %d,\n", r->_initial_code_size);
    fprintf(fp, "        \"consts_capacity\": %d,\n", r->_initial_consts_capacity);
    fprintf(fp, "        \"stubs_size\": %d,\n", r->_initial_stubs_size);
    fprintf(fp, "        \"locs_size\": %d\n", r->_initial_locs_size);
    fprintf(fp, "      },\n");
    fprintf(fp, "      \"expand_count\": %d,\n", r->_expand_count);
    fprintf(fp, "      \"total_expand_bytes\": %d,\n", r->_total_expand_bytes);
    fprintf(fp, "      \"expansions\": [");
    for (int j = 0; j < r->_expansions.length(); j++) {
      ExpansionEvent& e = r->_expansions.at(j);
      const char* sect_name = (e._section == 0) ? "consts"
                              : (e._section == 1) ? "insts" : "stubs";
      fprintf(fp, "%s\n        {\"section\": \"%s\", \"amount\": %d, \"new_total_cap\": %d, \"elapsed_us\": " JLONG_FORMAT "}",
              (j > 0 ? "," : ""), sect_name, e._amount, e._new_total_cap, e._elapsed_us);
    }
    fprintf(fp, "%s],\n", r->_expansions.length() > 0 ? "\n      " : "");
    fprintf(fp, "      \"final\": {\n");
    fprintf(fp, "        \"consts_usage\": %d,\n", r->_final_consts_size);
    fprintf(fp, "        \"consts_capacity\": %d,\n", r->_final_consts_capacity);
    fprintf(fp, "        \"consts_slack\": %d,\n", slack);
    fprintf(fp, "        \"consts_waste_rate\": %.4f,\n", waste);
    fprintf(fp, "        \"insts_size\": %d,\n", r->_final_insts_size);
    fprintf(fp, "        \"stubs_size\": %d\n", r->_final_stubs_size);
    fprintf(fp, "      },\n");
    fprintf(fp, "      \"finalize_latency_us\": " JLONG_FORMAT "\n", r->_finalize_latency_us);
    fprintf(fp, "    }%s\n", (i < _records.length() - 1) ? "," : "");
  }

  fprintf(fp, "  ]\n");
  fprintf(fp, "}\n");
  fclose(fp);

  // Summary
  tty->print_cr("=== CodeBuffer Instrumentation Summary ===");
  tty->print_cr("  Total compilations: %d", _records.length());
  int total_expands = 0;
  int total_expand_bytes = 0;
  jlong total_latency = 0;
  int total_slack = 0;
  int total_cap = 0;
  for (int i = 0; i < _records.length(); i++) {
    Record* r = _records.at(i);
    total_expands += r->_expand_count;
    total_expand_bytes += r->_total_expand_bytes;
    total_latency += r->_finalize_latency_us;
    total_slack += r->consts_slack();
    total_cap += r->_final_consts_capacity;
  }
  tty->print_cr("  Total expansions: %d", total_expands);
  tty->print_cr("  Total expand bytes: %d", total_expand_bytes);
  tty->print_cr("  Total finalize latency: " JLONG_FORMAT " us", total_latency);
  tty->print_cr("  Total consts slack: %d bytes", total_slack);
  tty->print_cr("  Total consts capacity: %d bytes", total_cap);
  if (total_cap > 0) {
    tty->print_cr("  Overall waste rate: %.2f%%", 100.0 * (double)total_slack / (double)total_cap);
  }
  tty->print_cr("  JSON output: %s", path);
}

void CodeBufferInstrumentation::output_and_shutdown() {
  if (!JeandleCodeBufferInstrument) return;
  instance()->output_json();
}

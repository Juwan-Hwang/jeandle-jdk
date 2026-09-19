#include "precompiled.hpp"
#include "asm/codeBufferInstrumentation.hpp"
#include "runtime/globals_shared.hpp"
#include "compiler/compiler_globals.hpp"
#include "utilities/ostream.hpp"
#include "runtime/os.hpp"

CodeBufferInstrumentation* CodeBufferInstrumentation::_instance = nullptr;

// ---- ConstLayoutStats ----

volatile jint  ConstLayoutStats::_methods_seen           = 0;
volatile jint  ConstLayoutStats::_exact_path             = 0;
volatile jint  ConstLayoutStats::_legacy_path            = 0;
volatile jint  ConstLayoutStats::_plan_failures          = 0;
volatile jint  ConstLayoutStats::_fallback_triggered     = 0;
volatile jint  ConstLayoutStats::_const_expand_after_plan = 0;
volatile jlong ConstLayoutStats::_exact_consts_bytes     = 0;
volatile jlong ConstLayoutStats::_legacy_consts_bytes    = 0;

void ConstLayoutStats::reset() {
  Atomic::store(&_methods_seen, (jint)0);
  Atomic::store(&_exact_path, (jint)0);
  Atomic::store(&_legacy_path, (jint)0);
  Atomic::store(&_plan_failures, (jint)0);
  Atomic::store(&_fallback_triggered, (jint)0);
  Atomic::store(&_const_expand_after_plan, (jint)0);
  Atomic::store(&_exact_consts_bytes, (jlong)0);
  Atomic::store(&_legacy_consts_bytes, (jlong)0);
}

void ConstLayoutStats::print_on(outputStream* st) {
  jlong exact  = exact_consts_bytes();
  jlong legacy = legacy_consts_bytes();
  st->print_cr("=== Deterministic Const Layout Summary ===");
  st->print_cr("  Methods reaching finalize:            %d", methods_seen());
  st->print_cr("  Exact path (from ConstSectionPlan):   %d", exact_path());
  st->print_cr("  Legacy path (48 KiB default):         %d", legacy_path());
  st->print_cr("  Planner failures:                     %d", plan_failures());
  st->print_cr("  Runtime layout fallbacks:             %d", fallback_triggered());
  st->print_cr("  consts expansions after exact plan:   %d", const_expand_after_plan());
  st->print_cr("  Consts bytes requested by planner:    " JLONG_FORMAT, exact);
  st->print_cr("  Consts bytes legacy would request:    " JLONG_FORMAT, legacy);
  if (legacy > 0) {
    st->print_cr("  Consts bytes saved:                   " JLONG_FORMAT " (%.2f%%)",
                 legacy - exact,
                 100.0 * (double)(legacy - exact) / (double)legacy);
  }
}

// ---- CodeBufferInstrumentation ----

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

  // Section 0 is SECT_CONSTS. An expansion there while running the exact
  // allocation strategy is exactly the degradation we promised to make visible
  // rather than hide: record it as an aggregate so it cannot be lost in the
  // per-method noise.
  if (section == 0 && r->_used_exact_allocation) {
    ConstLayoutStats::record_const_expand_after_plan();
  }
}

void CodeBufferInstrumentation::record_finalize(int consts_size, int consts_cap,
                                                 int insts_size, int stubs_size,
                                                 jlong finalize_latency_us,
                                                 int planned_consts_size,
                                                 int planned_padding,
                                                 int planned_alignment,
                                                 int const_entries,
                                                 const char* plan_status,
                                                 bool used_exact_allocation,
                                                 int layout_fallback_count) {
  if (!JeandleCodeBufferInstrument) return;
  CodeBufferInstrumentation* inst = instance();
  if (inst->_records.is_empty()) return;
  Record* r = inst->_records.at(inst->current_index());
  r->_final_consts_size = consts_size;
  r->_final_consts_capacity = consts_cap;
  r->_final_insts_size = insts_size;
  r->_final_stubs_size = stubs_size;
  r->_finalize_latency_us = finalize_latency_us;
  r->_planned_consts_size = planned_consts_size;
  r->_planned_padding = planned_padding;
  r->_planned_alignment = planned_alignment;
  r->_const_entries = const_entries;
  r->_plan_status = plan_status;
  r->_used_exact_allocation = used_exact_allocation;
  r->_layout_fallback_count = layout_fallback_count;
}

CodeBufferInstrumentation::Summary CodeBufferInstrumentation::compute_summary() const {
  Summary s = {};
  s.max_latency = -1;
  for (int i = 0; i < _records.length(); i++) {
    const Record* r = _records.at(i);
    s.total_records++;

    // A section that was never given a capacity is treated as "not allocated"
    // rather than "zero", so counts stay comparable between strategies.
    if (r->_final_consts_capacity > 0) {
      s.consts_allocating++;
      s.consts_capacity += r->_final_consts_capacity;
      s.consts_usage += r->_final_consts_size;
    }
    if (r->_planned_consts_size >= 0) {
      s.planned_size_sum += r->_planned_consts_size;
    }
    if (r->_planned_consts_size > 0) {
      s.nonempty_consts++;
    } else {
      s.zero_consts++;
    }
    if (r->_used_exact_allocation) {
      s.exact_path++;
    } else if (r->_planned_consts_size < 0) {
      s.legacy_path++;
    }
    if (r->_layout_fallback_count > 0) {
      s.fallback_records++;
    }

    s.total_expands += r->_expand_count;
    s.total_expand_bytes += r->_total_expand_bytes;
    for (int j = 0; j < r->_expansions.length(); j++) {
      const auto& e = r->_expansions.at(j);
      if (e._section == 0)      s.consts_expands++;
      else if (e._section == 1) s.insts_expands++;
      else                      s.stubs_expands++;
    }

    s.total_latency += r->_finalize_latency_us;
    if (r->_finalize_latency_us > s.max_latency) {
      s.max_latency = r->_finalize_latency_us;
    }
  }
  return s;
}

void CodeBufferInstrumentation::output_json() {
  if (_output_done) return;
  _output_done = true;

  Summary s = compute_summary();

  const char* path = "/tmp/codebuffer_instrument.json";
  FILE* fp = os::fopen(path, "w");
  if (fp == nullptr) {
    tty->print_cr("CodeBufferInstrumentation: failed to open %s", path);
    return;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"instrumentation\": \"codebuffer\",\n");
  fprintf(fp, "  \"counting_unit\": \"recorded CodeBuffer initialize()\",\n");
  fprintf(fp, "  \"summary\": {\n");
  fprintf(fp, "    \"total_records\": %d,\n", s.total_records);
  fprintf(fp, "    \"consts_allocating_records\": %d,\n", s.consts_allocating);
  fprintf(fp, "    \"records_with_const_sections\": %d,\n", s.nonempty_consts);
  fprintf(fp, "    \"records_without_const_sections\": %d,\n", s.zero_consts);
  fprintf(fp, "    \"exact_path_records\": %d,\n", s.exact_path);
  fprintf(fp, "    \"legacy_path_records\": %d,\n", s.legacy_path);
  fprintf(fp, "    \"fallback_records\": %d,\n", s.fallback_records);
  fprintf(fp, "    \"expand\": {\n");
  fprintf(fp, "      \"total\": %d,\n", s.total_expands);
  fprintf(fp, "      \"consts\": %d,\n", s.consts_expands);
  fprintf(fp, "      \"insts\": %d,\n", s.insts_expands);
  fprintf(fp, "      \"stubs\": %d,\n", s.stubs_expands);
  fprintf(fp, "      \"total_bytes\": %d\n", s.total_expand_bytes);
  fprintf(fp, "    },\n");
  fprintf(fp, "    \"consts_usage_bytes\": %d,\n", s.consts_usage);
  fprintf(fp, "    \"consts_capacity_bytes\": %d,\n", s.consts_capacity);
  fprintf(fp, "    \"planned_consts_bytes\": %d,\n", s.planned_size_sum);
  fprintf(fp, "    \"finalize_latency_us_total\": " JLONG_FORMAT ",\n", s.total_latency);
  fprintf(fp, "    \"finalize_latency_us_max\": " JLONG_FORMAT ",\n", s.max_latency);
  fprintf(fp, "    \"consts_waste_rate\": %.6f\n",
          s.consts_capacity > 0
            ? (double)(s.consts_capacity - s.consts_usage) / (double)s.consts_capacity
            : 0.0);
  fprintf(fp, "  },\n");
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
    fprintf(fp, "      \"plan\": {\n");
    fprintf(fp, "        \"status\": \"%s\",\n", r->_plan_status ? r->_plan_status : "");
    fprintf(fp, "        \"entries\": %d,\n", r->_const_entries);
    fprintf(fp, "        \"planned_consts_size\": %d,\n", r->_planned_consts_size);
    fprintf(fp, "        \"planned_padding\": %d,\n", r->_planned_padding);
    fprintf(fp, "        \"planned_alignment\": %d,\n", r->_planned_alignment);
    fprintf(fp, "        \"used_exact_allocation\": %s,\n",
            r->_used_exact_allocation ? "true" : "false");
    fprintf(fp, "        \"layout_fallback_count\": %d\n", r->_layout_fallback_count);
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

  tty->print_cr("=== CodeBuffer Instrumentation Summary ===");
  tty->print_cr("  Counting unit: recorded CodeBuffer initialize()");
  tty->print_cr("  Total records: %d", s.total_records);
  tty->print_cr("    consts section allocated:     %d", s.consts_allocating);
  tty->print_cr("    with const sections:          %d", s.nonempty_consts);
  tty->print_cr("    without const sections:       %d", s.zero_consts);
  tty->print_cr("  Expand: total %d (consts %d, insts %d, stubs %d), bytes %d",
                s.total_expands, s.consts_expands, s.insts_expands, s.stubs_expands,
                s.total_expand_bytes);
  tty->print_cr("  Consts: usage %d bytes / capacity %d bytes, waste %.2f%%",
                s.consts_usage, s.consts_capacity,
                s.consts_capacity > 0
                  ? 100.0 * (double)(s.consts_capacity - s.consts_usage) / (double)s.consts_capacity
                  : 0.0);
  tty->print_cr("  Finalize latency: total " JLONG_FORMAT " us, max " JLONG_FORMAT " us",
                s.total_latency, s.max_latency);
  tty->print_cr("  JSON output: %s", path);

  ConstLayoutStats::print_on(tty);
}

void CodeBufferInstrumentation::output_and_shutdown() {
  if (!JeandleCodeBufferInstrument) return;
  instance()->output_json();
}

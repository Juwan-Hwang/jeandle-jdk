#include "precompiled.hpp"
#include "asm/codeBufferInstrumentation.hpp"
#include "runtime/mutexLocker.hpp"
#include "runtime/globals_shared.hpp"
#include "compiler/compiler_globals.hpp"
#include "utilities/ostream.hpp"
#include "runtime/os.hpp"

// Week 9: the locs census is read straight off the CodeBuffer, so this file needs the
// full definitions. codeBuffer.hpp includes this header, which is fine here (a .cpp may
// depend in both directions; a header must not).
#ifndef PRODUCT
#include "asm/codeBuffer.hpp"
#include "code/relocInfo.hpp"

// relocInfo type names, in the order of relocInfo::relocType. Kept next to the census so
// a histogram row can never be printed with the wrong label.
const char* const codebuffer_reloc_type_names[] = {
  "none",
  "oop",
  "virtual_call",
  "opt_virtual_call",
  "static_call",
  "static_stub",
  "runtime_call",
  "external_word",
  "internal_word",
  "section_word",
  "poll",
  "poll_return",
  "metadata",
  "trampoline_stub",
  "runtime_call_w_cp",
  "data_prefix_tag",
  "post_call_nop",
  "entry_guard",
  "barrier",
  "jeandle_section_word",
  "jeandle_oop",
  "jeandle_oop_addr",
};
const int codebuffer_reloc_type_name_count =
    (int)(sizeof(codebuffer_reloc_type_names) / sizeof(codebuffer_reloc_type_names[0]));

// Guards the table above against an upstream relocInfo change: if a type is inserted or
// renamed, this fires on the first census read instead of silently mislabelling rows.
static const char* reloc_type_name(int kind) {
  if (kind < 0 || kind >= codebuffer_reloc_type_name_count) {
    return "unknown";
  }
  return codebuffer_reloc_type_names[kind];
}
#endif // PRODUCT

std::atomic<CodeBufferInstrumentation*> CodeBufferInstrumentation::_instance(nullptr);

// ---- ConstLayoutStats ----

volatile jint  ConstLayoutStats::_methods_seen           = 0;
volatile jint  ConstLayoutStats::_exact_path             = 0;
volatile jint  ConstLayoutStats::_legacy_path            = 0;
volatile jint  ConstLayoutStats::_plan_failures          = 0;
volatile jint  ConstLayoutStats::_fallback_triggered     = 0;
volatile jint  ConstLayoutStats::_const_expand_after_plan = 0;
volatile jint  ConstLayoutStats::_locs_expand_after_plan   = 0;
volatile jlong ConstLayoutStats::_exact_consts_bytes     = 0;
volatile jlong ConstLayoutStats::_legacy_consts_bytes    = 0;

void ConstLayoutStats::reset() {
  Atomic::store(&_methods_seen, (jint)0);
  Atomic::store(&_exact_path, (jint)0);
  Atomic::store(&_legacy_path, (jint)0);
  Atomic::store(&_plan_failures, (jint)0);
  Atomic::store(&_fallback_triggered, (jint)0);
  Atomic::store(&_const_expand_after_plan, (jint)0);
  Atomic::store(&_locs_expand_after_plan, (jint)0);
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
  st->print_cr("  locs expansions after exact plan:     %d", locs_expand_after_plan());
  st->print_cr("  Consts bytes requested by planner:    " JLONG_FORMAT, exact);
  st->print_cr("  Consts bytes legacy would request:    " JLONG_FORMAT, legacy);
  if (legacy > 0) {
    st->print_cr("  Consts bytes saved:                   " JLONG_FORMAT " (%.2f%%)",
                 legacy - exact,
                 100.0 * (double)(legacy - exact) / (double)legacy);
  }
}

// ---- CodeBufferInstrumentation ----

// The instance pointer is published atomically: the first compiler threads to
// arrive must not each build their own.
bool CodeBufferInstrumentation::enabled() {
#ifndef PRODUCT
  return JeandleCodeBufferInstrument;
#else
  return false;
#endif
}

const char* CodeBufferInstrumentation::output_path(char* buf, size_t buflen) {
#ifndef PRODUCT
  const char* pattern = JeandleCodeBufferInstrumentFile;
  const char* p = strchr(pattern, '%');
  if (p != nullptr && p[1] == 'p') {
    // Same convention as -XX:ErrorFile: %p expands to the process id, so
    // concurrent JVMs and repeated runs cannot overwrite each other.
    char prefix[256];
    size_t len = (size_t)(p - pattern);
    if (len >= sizeof(prefix)) {
      len = sizeof(prefix) - 1;
    }
    memcpy(prefix, pattern, len);
    prefix[len] = '\0';
    os::snprintf(buf, buflen, "%s%d%s", prefix, os::current_process_id(), p + 2);
    return buf;
  }
  os::snprintf(buf, buflen, "%s", pattern);
  return buf;
#else
  buf[0] = '\0';
  return buf;
#endif
}

CodeBufferInstrumentation* CodeBufferInstrumentation::instance() {
  CodeBufferInstrumentation* result = _instance.load(std::memory_order_acquire);
  if (result == nullptr) {
    CodeBufferInstrumentation* candidate = new CodeBufferInstrumentation();
    CodeBufferInstrumentation* expected = nullptr;
    if (_instance.compare_exchange_strong(expected, candidate, std::memory_order_release)) {
      result = candidate;
    } else {
      delete candidate;
      result = expected;
    }
  }
  return result;
}

// Scanned from the end because a buffer is recorded and then used immediately; a
// -Xcomp run holds a few thousand records, which is cheap for a facility that is
// off by default. Callers create a record when the lookup fails, so an event can
// never be lost - the old code could only mis-attribute it.
CodeBufferInstrumentation::Record* CodeBufferInstrumentation::find_locked(const CodeBuffer* cb) const {
  for (int i = _records.length() - 1; i >= 0; i--) {
    if (_records.at(i)->_buffer == cb && !_records.at(i)->_retired) {
      return _records.at(i);
    }
  }
  return nullptr;
}

void CodeBufferInstrumentation::record_retire(const CodeBuffer* cb) {
  if (!enabled()) return;
  CodeBufferInstrumentation* inst = instance();
  MutexLocker ml(&inst->_lock, Mutex::_no_safepoint_check_flag);
  Record* r = inst->find_locked(cb);
  if (r != nullptr) {
    r->_retired = true;
  }
}

void CodeBufferInstrumentation::record_initialize(const CodeBuffer* cb, const char* name, int code_size,
                                                   int consts_cap, int stubs_size,
                                                   int locs_size) {
  if (!enabled()) return;
  // Allocate before taking the lock: _records is written from several compiler
  // threads, but the lock must not be held across allocation.
  Record* fresh = new Record(name, code_size, consts_cap, stubs_size, locs_size);
  fresh->_buffer = cb;
  CodeBufferInstrumentation* inst = instance();
  {
    MutexLocker ml(&inst->_lock, Mutex::_no_safepoint_check_flag);
    Record* r = inst->find_locked(cb);
    if (r == nullptr) {
      inst->_records.append(fresh);
      return;
    }
    // A repeated initialize() for the same live buffer carries better numbers
    // (the delegating overload only sees the combined code size; Jeandle knows
    // the consts capacity). Keep the most informative value per field instead
    // of appending a duplicate record.
    if (code_size  > r->_initial_code_size)       r->_initial_code_size = code_size;
    if (consts_cap > r->_initial_consts_capacity) r->_initial_consts_capacity = consts_cap;
    if (stubs_size > r->_initial_stubs_size)      r->_initial_stubs_size = stubs_size;
    if (locs_size  > r->_initial_locs_size)       r->_initial_locs_size = locs_size;
  }
  delete fresh;
}

void CodeBufferInstrumentation::record_expand(const CodeBuffer* cb, int section, int amount,
                                               int new_total_cap, jlong elapsed_us) {
  if (!enabled()) return;
  // Allocated before the lock (see record_initialize) so that nothing that can
  // take another lock happens while _lock is held.
  Record* fresh = new Record("unknown", 0, 0, 0, 0);
  fresh->_buffer = cb;
  CodeBufferInstrumentation* inst = instance();
  Record* r;
  {
    MutexLocker ml(&inst->_lock, Mutex::_no_safepoint_check_flag);
    r = inst->find_locked(cb);
    if (r == nullptr) {
      r = fresh;
      inst->_records.append(r);
      fresh = nullptr;
    }
  }
  if (fresh != nullptr) {
    delete fresh;
  }
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

void CodeBufferInstrumentation::record_finalize(const CodeBuffer* cb,
                                                int consts_size, int consts_cap,
                                                 int insts_size, int stubs_size,
                                                 jlong finalize_latency_us,
                                                 int planned_consts_size,
                                                 int planned_padding,
                                                 int planned_alignment,
                                                 int const_entries,
                                                 const char* plan_status,
                                                 bool used_exact_allocation,
                                                 int layout_fallback_count) {
  if (!enabled()) return;
  // Allocated before the lock (see record_initialize) so that nothing that can
  // take another lock happens while _lock is held.
  Record* fresh = new Record("unknown", 0, 0, 0, 0);
  fresh->_buffer = cb;
  CodeBufferInstrumentation* inst = instance();
  Record* r;
  {
    MutexLocker ml(&inst->_lock, Mutex::_no_safepoint_check_flag);
    r = inst->find_locked(cb);
    if (r == nullptr) {
      r = fresh;
      inst->_records.append(r);
      fresh = nullptr;
    }
  }
  if (fresh != nullptr) {
    delete fresh;
  }
  r->_finalized = true;
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

void CodeBufferInstrumentation::record_locs(const CodeBuffer* cb, const int* planned_bytes) {
  if (!enabled()) return;
#ifndef PRODUCT
  // The census lives in the buffer, so a compilation is reported while its CodeBuffer is
  // still alive - the same lifetime rule that record_finalize() relies on.
  assert((int)CodeBuffer::SECT_LIMIT == (int)locs_sections, "section count drifted");
  assert(CodeBuffer::reloc_kind_count() <= (int)reloc_kind_max, "reloc type table too small");
  assert(codebuffer_reloc_type_name_count == (int)relocInfo::jeandle_oop_addr_type + 1,
         "reloc type name table drifted");

  Record* fresh = new Record("unknown", 0, 0, 0, 0);
  fresh->_buffer = cb;
  CodeBufferInstrumentation* inst = instance();
  Record* r;
  {
    MutexLocker ml(&inst->_lock, Mutex::_no_safepoint_check_flag);
    r = inst->find_locked(cb);
    if (r == nullptr) {
      r = fresh;
      inst->_records.append(r);
      fresh = nullptr;
    }
  }
  if (fresh != nullptr) {
    delete fresh;
  }

  for (int sec = 0; sec < (int)locs_sections; sec++) {
    const CodeSection* cs = cb->code_section(sec);
    r->_locs_planned_bytes[sec] = (planned_bytes != nullptr) ? planned_bytes[sec] : -1;
    r->_locs_initial_elements[sec] = cb->locs_initial_capacity(sec);
    r->_locs_used_elements[sec]    = cs->has_locs() ? cs->locs_count() : 0;
    r->_locs_final_elements[sec]   = cs->has_locs() ? cs->locs_capacity() : 0;
    r->_locs_expansions[sec]       = cb->locs_expand_count(sec);
    r->_locs_expand_bytes[sec]     = cb->locs_expand_bytes(sec);
    r->_locs_gap_fillers[sec]      = cb->locs_gap_fillers(sec);
    r->_locs_unstored[sec]         = cb->locs_unstored_relocs(sec);
    for (int k = 0; k < (int)reloc_kind_max; k++) {
      r->_reloc_records[sec][k]  = cb->reloc_records(sec, k);
      r->_reloc_elements[sec][k] = cb->reloc_elements(sec, k);
    }
  }
#endif // PRODUCT
}

CodeBufferInstrumentation::Summary CodeBufferInstrumentation::compute_summary() const {
  Summary s = {};
  s.max_latency = -1;
  for (int i = 0; i < _records.length(); i++) {
    const Record* r = _records.at(i);
    s.total_records++;
    if (r->_finalized) { s.finalized_records++; } else { s.unfinalized_records++; }

    // A section that was never given a capacity is treated as "not allocated"
    // rather than "zero", so counts stay comparable between strategies.
    // Only a finalized record describes a compilation that actually happened; a
    // temporary CodeBuffer from expand() must not inflate these totals.
    if (r->_finalized && r->_final_consts_capacity > 0) {
      s.consts_allocating++;
      s.consts_capacity += r->_final_consts_capacity;
      s.consts_usage += r->_final_consts_size;
    }
    if (r->_finalized && r->_planned_consts_size >= 0) {
      s.planned_size_sum += r->_planned_consts_size;
    }
    if (!r->_finalized) {
      // outside both populations; visible via unfinalized_records
    } else if (r->_planned_consts_size > 0) {
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

    // Week 9: the second growth path. Only finalized records describe a compilation that
    // happened, so the census of a temporary CodeBuffer is counted nowhere but here.
    if (r->_finalized) {
      int record_expansions = 0;
      for (int sec = 0; sec < (int)locs_sections; sec++) {
        s.locs_expansions    += r->_locs_expansions[sec];
        s.locs_expand_bytes  += r->_locs_expand_bytes[sec];
        s.locs_used_elements += r->_locs_used_elements[sec];
        s.locs_capacity_elements += r->_locs_final_elements[sec];
        s.locs_gap_fillers   += r->_locs_gap_fillers[sec];
        s.locs_unstored      += r->_locs_unstored[sec];
        record_expansions    += r->_locs_expansions[sec];
        if (r->_locs_planned_bytes[sec] >= 0) {
          s.locs_planned_bytes += r->_locs_planned_bytes[sec];
        }
        for (int k = 0; k < (int)reloc_kind_max; k++) {
          s.reloc_kind_records[k]  += r->_reloc_records[sec][k];
          s.reloc_kind_elements[k] += r->_reloc_elements[sec][k];
          s.reloc_records_total    += r->_reloc_records[sec][k];
          s.reloc_elements_total   += r->_reloc_elements[sec][k];
        }
      }
      if (record_expansions > 0) {
        s.records_with_locs_expansion++;
      }
      if (r->_locs_planned_bytes[1] >= 0) {
        s.records_with_exact_locs++;
      }
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

  char path_buf[512];
  const char* path = output_path(path_buf, sizeof(path_buf));
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
  // Week 9: the relocation array has its own growth path (CodeSection::expand_locs),
  // separate from the CodeCache-level expansion above. Elements are relocInfo slots.
  fprintf(fp, "    \"locs\": {\n");
  fprintf(fp, "      \"unit\": \"relocInfo elements (2 bytes each) unless suffixed _bytes\",\n");
  fprintf(fp, "      \"expansions\": %d,\n", s.locs_expansions);
  fprintf(fp, "      \"expansion_bytes\": %d,\n", s.locs_expand_bytes);
  fprintf(fp, "      \"used_elements\": %d,\n", s.locs_used_elements);
  fprintf(fp, "      \"capacity_elements\": %d,\n", s.locs_capacity_elements);
  fprintf(fp, "      \"planned_bytes\": %d,\n", s.locs_planned_bytes);
  fprintf(fp, "      \"gap_fillers\": %d,\n", s.locs_gap_fillers);
  fprintf(fp, "      \"unstored_relocs\": %d,\n", s.locs_unstored);
  fprintf(fp, "      \"records_with_expansion\": %d,\n", s.records_with_locs_expansion);
  fprintf(fp, "      \"records_with_exact_plan\": %d\n", s.records_with_exact_locs);
  fprintf(fp, "    },\n");
  fprintf(fp, "    \"reloc_total_records\": %d,\n", s.reloc_records_total);
  fprintf(fp, "    \"reloc_total_elements\": %d,\n", s.reloc_elements_total);
#ifndef PRODUCT
  fprintf(fp, "    \"reloc_distribution\": [\n");
  // One row per relocInfo type that any finalized compilation actually received.
  bool first_kind = true;
  for (int k = 0; k < (int)reloc_kind_max; k++) {
    if (s.reloc_kind_records[k] == 0) {
      continue;
    }
    fprintf(fp, "      %s{\"kind\": \"%s\", \"records\": %d, \"elements\": %d, \"mean_elements\": %.3f}",
            first_kind ? "" : ",\n      ", reloc_type_name(k),
            s.reloc_kind_records[k], s.reloc_kind_elements[k],
            (double)s.reloc_kind_elements[k] / (double)s.reloc_kind_records[k]);
    first_kind = false;
  }
  fprintf(fp, "%s\n    ],\n", first_kind ? "" : "");
#endif // PRODUCT
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
    fprintf(fp, "      \"finalized\": %s,\n", r->_finalized ? "true" : "false");
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
    // Week 9: per-section relocation array census. planned_bytes < 0 means the locs model
    // did not size this section (legacy heuristic, or the whole buffer took the fallback).
    static const char* const locs_section_names[locs_sections] = { "consts", "insts", "stubs" };
    fprintf(fp, "      \"locs\": {\n");
    for (int sec = 0; sec < (int)locs_sections; sec++) {
      int records = 0;
      int elements = 0;
      for (int k = 0; k < (int)reloc_kind_max; k++) {
        records  += r->_reloc_records[sec][k];
        elements += r->_reloc_elements[sec][k];
      }
      fprintf(fp, "        \"%s\": {\"planned_bytes\": %d, \"initial_elements\": %d, \"used_elements\": %d, \"capacity_elements\": %d, \"expansions\": %d, \"expansion_bytes\": %d, \"gap_fillers\": %d, \"unstored_relocs\": %d, \"records\": %d, \"elements\": %d}%s\n",
              locs_section_names[sec], r->_locs_planned_bytes[sec],
              r->_locs_initial_elements[sec], r->_locs_used_elements[sec],
              r->_locs_final_elements[sec], r->_locs_expansions[sec],
              r->_locs_expand_bytes[sec], r->_locs_gap_fillers[sec],
              r->_locs_unstored[sec], records, elements,
              (sec < (int)locs_sections - 1) ? "," : "");
    }
    fprintf(fp, "      },\n");
    fprintf(fp, "      \"finalize_latency_us\": " JLONG_FORMAT "\n", r->_finalize_latency_us);
    fprintf(fp, "    }%s\n", (i < _records.length() - 1) ? "," : "");
  }

  fprintf(fp, "  ]\n");
  fprintf(fp, "}\n");
  fclose(fp);

  tty->print_cr("=== CodeBuffer Instrumentation Summary ===");
  tty->print_cr("  Counting unit: recorded CodeBuffer initialize()");
  tty->print_cr("  Total records: %d (finalized %d, temporary/unfinalized %d)",
                s.total_records, s.finalized_records, s.unfinalized_records);
  tty->print_cr("    consts section allocated:     %d", s.consts_allocating);
  tty->print_cr("    with const sections:          %d", s.nonempty_consts);
  tty->print_cr("    without const sections:       %d", s.zero_consts);
  tty->print_cr("  Expand: total %d (consts %d, insts %d, stubs %d), bytes %d",
                s.total_expands, s.consts_expands, s.insts_expands, s.stubs_expands,
                s.total_expand_bytes);
  tty->print_cr("  Locs expand (separate growth path): %d reallocations, %d bytes added,"
                " %d records affected",
                s.locs_expansions, s.locs_expand_bytes,
                s.records_with_locs_expansion);
  tty->print_cr("  Relocs: %d records / %d elements, gap fillers %d, unstored %d",
                s.reloc_records_total, s.reloc_elements_total,
                s.locs_gap_fillers, s.locs_unstored);
  tty->print_cr("  Locs elements: used %d / capacity %d (planned %d bytes)",
                s.locs_used_elements, s.locs_capacity_elements, s.locs_planned_bytes);
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
  if (!enabled()) return;
  instance()->output_json();
}

#ifndef SHARE_ASM_CODEBUFFERINSTRUMENTATION_HPP
#define SHARE_ASM_CODEBUFFERINSTRUMENTATION_HPP

#include "utilities/globalDefinitions.hpp"
#include "utilities/ticks.hpp"
#include "utilities/growableArray.hpp"
#include "runtime/atomic.hpp"
#include <atomic>

class outputStream;

class CodeBufferInstrumentation : public CHeapObj<mtCompiler> {
 private:
  struct ExpansionEvent {
    int  _section;        // SECT_CONSTS=0, SECT_INSTS=1, SECT_STUBS=2
    int  _amount;         // requested expansion bytes
    int  _new_total_cap;  // new total capacity after expansion
    jlong _elapsed_us;    // microseconds for this expansion
  };

  // Per-compilation record with full telemetry
  struct Record {
    const char* _name;
    // Initial allocation (from CodeBuffer::initialize + initialize_consts_size)
    int  _initial_code_size;       // total code buffer size requested
    int  _initial_consts_capacity; // consts section pre-allocated capacity
    int  _initial_stubs_size;      // stubs section size
    int  _initial_locs_size;       // relocation locs size
    // Expansion tracking
    int  _expand_count;
    GrowableArrayCHeap<ExpansionEvent, mtCompiler> _expansions;
    int  _total_expand_bytes;      // sum of all expansion amounts
    // Final actual usage (from CodeBuffer sections after finalize)
    int  _final_consts_size;       // actual bytes used in consts section
    int  _final_insts_size;        // actual bytes used in insts section
    int  _final_stubs_size;        // actual bytes used in stubs section
    int  _final_consts_capacity;   // final capacity of consts section
    // Timing
    jlong _finalize_latency_us;    // finalize() elapsed time in microseconds
    // Week 5: deterministic const layout telemetry
    // _planned_consts_size < 0 means the planner was not used (legacy allocation).
    int  _planned_consts_size;     // ConstSectionPlan::total_size()
    int  _planned_padding;         // ConstSectionPlan::total_padding()
    int  _planned_alignment;       // ConstSectionPlan::max_alignment()
    int  _const_entries;           // number of planned const sections
    const char* _plan_status;      // ConstSectionPlan::status_name()
    bool _used_exact_allocation;   // consts capacity came from the plan, not the 48 KiB default
    int  _layout_fallback_count;   // runtime discovery of an unplanned const section

    Record(const char* name, int code_size, int consts_cap,
           int stubs_size, int locs_size)
      : _name(name), _initial_code_size(code_size),
        _initial_consts_capacity(consts_cap), _initial_stubs_size(stubs_size),
        _initial_locs_size(locs_size), _expand_count(0), _expansions(2),
        _total_expand_bytes(0), _final_consts_size(0), _final_insts_size(0),
        _final_stubs_size(0), _final_consts_capacity(0), _finalize_latency_us(0),
        _planned_consts_size(-1), _planned_padding(0), _planned_alignment(0),
        _const_entries(0), _plan_status("Unplanned"),
        _used_exact_allocation(false), _layout_fallback_count(0) {}

    // Computed metrics
    int consts_slack() const { return _final_consts_capacity - _final_consts_size; }
    double consts_waste_rate() const {
      return _final_consts_capacity > 0
        ? (double)consts_slack() / (double)_final_consts_capacity
        : 0.0;
    }
  };

  static CodeBufferInstrumentation* _instance;
  GrowableArrayCHeap<Record*, mtCompiler> _records;
  bool _output_done;

  // One unambiguous counting unit for every number this tool reports.
  //
  // Week 1-4 mixed three different populations ("methods compiled" from the
  // plan trace, "records" from this instrumentation, and a capacity-derived
  // count), which is how 89 / 72 / 17 ended up side by side in one report.
  // Every number printed now derives from exactly this struct, over exactly one
  // population: the recorded initialize() calls.
  struct Summary {
    int total_records;        // every recorded initialize() -> the counting unit
    int consts_allocating;    // records that actually reserved a consts section
    int nonempty_consts;      // records planning at least one const section
    int zero_consts;          // records planning none
    int exact_path;           // consts capacity came from ConstSectionPlan
    int legacy_path;          // consts capacity came from the 48 KiB default
    int fallback_records;     // records that hit runtime layout fallback
    int total_expands;
    int consts_expands;
    int insts_expands;
    int stubs_expands;
    int total_expand_bytes;
    int consts_usage;
    int consts_capacity;
    int planned_size_sum;
    jlong total_latency;
    jlong max_latency;
  };

  Summary compute_summary() const;

  CodeBufferInstrumentation() : _records(8), _output_done(false) {}
  void output_json();

 public:
  static CodeBufferInstrumentation* instance();

  void record_initialize(const char* name, int code_size, int consts_cap,
                         int stubs_size, int locs_size);

  void record_expand(int section, int amount, int new_total_cap, jlong elapsed_us);

  // Enhanced finalize: captures actual usage + capacity + latency + plan telemetry.
  // planned_consts_size < 0 => planner not used (legacy path).
  void record_finalize(int consts_size, int consts_cap, int insts_size,
                       int stubs_size, jlong finalize_latency_us,
                       int planned_consts_size, int planned_padding,
                       int planned_alignment, int const_entries,
                       const char* plan_status, bool used_exact_allocation,
                       int layout_fallback_count);

  static void output_and_shutdown();
  int current_index() const { return _records.length() - 1; }
};

// Aggregate counters for the deterministic const layout work.
//
// These are global and updated from Jeandle compiler threads, so every field is
// updated atomically. They exist so that Week 5 can answer "how many methods
// actually took the exact path, and how many had to degrade?" without having to
// post-process per-method records by hand.
class ConstLayoutStats : public AllStatic {
 private:
  static volatile jint _methods_seen;        // methods that reached finalize()
  static volatile jint _exact_path;          // allocated from ConstSectionPlan
  static volatile jint _legacy_path;         // allocated with the 48 KiB default
  static volatile jint _plan_failures;       // planner refused to produce a layout
  static volatile jint _fallback_triggered;  // runtime discovery of an unplanned section
  static volatile jint _const_expand_after_plan; // consts expanded despite exact allocation
  static volatile jlong _exact_consts_bytes;  // consts bytes handed out by the planner
  static volatile jlong _legacy_consts_bytes; // consts bytes that legacy would have used

 public:
  static void reset();

  static void record_method() { Atomic::add(&_methods_seen, (jint)1); }

  static void record_exact(size_t consts_bytes) {
    Atomic::add(&_exact_path, (jint)1);
    Atomic::add(&_exact_consts_bytes, (jlong)consts_bytes);
  }

  static void record_legacy(size_t consts_bytes) {
    Atomic::add(&_legacy_path, (jint)1);
    Atomic::add(&_legacy_consts_bytes, (jlong)consts_bytes);
  }

  static void record_plan_failure() { Atomic::add(&_plan_failures, (jint)1); }
  static void record_fallback()     { Atomic::add(&_fallback_triggered, (jint)1); }
  static void record_const_expand_after_plan() {
    Atomic::add(&_const_expand_after_plan, (jint)1);
  }

  static int  methods_seen()               { return Atomic::load(&_methods_seen); }
  static int  exact_path()                 { return Atomic::load(&_exact_path); }
  static int  legacy_path()                { return Atomic::load(&_legacy_path); }
  static int  plan_failures()              { return Atomic::load(&_plan_failures); }
  static int  fallback_triggered()         { return Atomic::load(&_fallback_triggered); }
  static int  const_expand_after_plan()    { return Atomic::load(&_const_expand_after_plan); }
  static jlong exact_consts_bytes()        { return Atomic::load(&_exact_consts_bytes); }
  static jlong legacy_consts_bytes()       { return Atomic::load(&_legacy_consts_bytes); }

  static void print_on(outputStream* st);
};

// RAII timer
class CodeBufferTimer {
 private:
  Ticks _start;
 public:
  CodeBufferTimer() { _start.stamp(); }
  jlong elapsed_us() const {
    Tickspan span = Ticks::now() - _start;
    return (jlong)(span.seconds() * 1000000.0);
  }
};

#endif // SHARE_ASM_CODEBUFFERINSTRUMENTATION_HPP

#ifndef SHARE_ASM_CODEBUFFERINSTRUMENTATION_HPP
#define SHARE_ASM_CODEBUFFERINSTRUMENTATION_HPP

#include "utilities/globalDefinitions.hpp"
#include "utilities/ticks.hpp"
#include "utilities/growableArray.hpp"
#include <atomic>

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

    Record(const char* name, int code_size, int consts_cap,
           int stubs_size, int locs_size)
      : _name(name), _initial_code_size(code_size),
        _initial_consts_capacity(consts_cap), _initial_stubs_size(stubs_size),
        _initial_locs_size(locs_size), _expand_count(0), _expansions(2),
        _total_expand_bytes(0), _final_consts_size(0), _final_insts_size(0),
        _final_stubs_size(0), _final_consts_capacity(0), _finalize_latency_us(0) {}

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

  CodeBufferInstrumentation() : _records(8), _output_done(false) {}
  void output_json();

 public:
  static CodeBufferInstrumentation* instance();

  void record_initialize(const char* name, int code_size, int consts_cap,
                         int stubs_size, int locs_size);

  void record_expand(int section, int amount, int new_total_cap, jlong elapsed_us);

  // Enhanced finalize: captures actual usage + capacity + latency
  void record_finalize(int consts_size, int consts_cap, int insts_size,
                       int stubs_size, jlong finalize_latency_us);

  static void output_and_shutdown();
  int current_index() const { return _records.length() - 1; }
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

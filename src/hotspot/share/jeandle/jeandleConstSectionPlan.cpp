/*
 * Copyright (c) 2025, the Jeandle-JDK Authors. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

#if defined(__has_include) && __has_include("jeandle/jeandleConstSectionPlan.hpp")
#include "jeandle/jeandleConstSectionPlan.hpp"
#else
#include "jeandleConstSectionPlan.hpp"
#endif

#include <algorithm>

bool ConstSectionPlan::calculate_layout() {
  if (_entries.empty()) {
    _total_size = 0;
    _max_alignment = 1;
    _is_valid = true;
    return true;
  }

  // Step 1: Canonical sorting according to the selected policy
  if (_policy == ConstSortPolicy::AlignmentDescending) {
    // Canonical Axioms:
    // Primary: alignment descending (minimizes/eliminates internal padding)
    // Secondary: section name ascending (deterministic tie-break across builds)
    // Tertiary: section index ascending (stable physical tie-break)
    std::sort(_entries.begin(), _entries.end(),
      [](const ConstSectionPlanEntry& a, const ConstSectionPlanEntry& b) {
        if (a._alignment != b._alignment) {
          return a._alignment > b._alignment;
        }
        if (a._section_name != b._section_name) {
          return a._section_name < b._section_name;
        }
        return a._section_index < b._section_index;
      });
  } else {
    // Baseline / Physical Order:
    // Sort by ELF section index ascending
    std::sort(_entries.begin(), _entries.end(),
      [](const ConstSectionPlanEntry& a, const ConstSectionPlanEntry& b) {
        return a._section_index < b._section_index;
      });
  }

  // Step 2: Deterministic cursor accumulation and alignment validation
  uint64_t cursor = 0;
  uint64_t max_align = 1;

  for (auto& entry : _entries) {
    uint64_t align = entry._alignment;

    // Strict boundary checks:
    // 1. Non-zero
    // 2. Power of 2
    // 3. Within MAX_SUPPORTED_CONST_ALIGNMENT contract (64 bytes)
    if (align == 0 || (align & (align - 1)) != 0 || align > MAX_SUPPORTED_CONST_ALIGNMENT) {
      _is_valid = false;
      return false;
    }

    if (align > max_align) {
      max_align = align;
    }

    // Advance cursor to aligned position
    uint64_t padding = (align - (cursor & (align - 1))) & (align - 1);
    if (UINT64_MAX - cursor < padding) {
      _is_valid = false;
      return false; // Overflow in padding
    }
    cursor += padding;

    entry._consts_offset = cursor;

    // Advance cursor by size with robust overflow check
    if (UINT64_MAX - cursor < entry._size) {
      _is_valid = false;
      return false; // Overflow in section size accumulation
    }
    cursor += entry._size;
  }

  _total_size = cursor;
  _max_alignment = max_align;
  _is_valid = true;
  return true;
}

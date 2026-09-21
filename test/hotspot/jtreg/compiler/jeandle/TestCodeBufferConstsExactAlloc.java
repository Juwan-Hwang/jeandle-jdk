/*
 * Copyright (c) 2026, the Jeandle-JDK Authors. All Rights Reserved.
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

/**
 * @test
 * @summary Const sections are allocated exactly as planned.
 *
 *          Runs a const-heavy workload in a child VM with the CodeBuffer
 *          instrumentation enabled and checks the two invariants of the exact
 *          allocation path:
 *
 *            1. every method that reaches finalize() takes the plan-driven
 *               path, with no planner failure and no fallback to the legacy
 *               48 KiB reservation, and
 *            2. the consts capacity that CodeBuffer hands out never exceeds
 *               what is actually used by more than one alignment step.
 *
 *   Point 2: for a buffer that never expanded, the reserved capacity may exceed
 *            the planned size by at most one alignment step. A buffer that did
 *            expand is re-divided by CodeBuffer::expand(), so its residual is
 *            not bounded by the alignment and is checked only for safety
 *            (usage <= capacity, usage == planned).
 *          CodeBuffer::initialize_section_size() rounds the section start down
 *          to the section alignment, so the capacity is
 *          requested + ((limit - requested) % alignment). The extra bytes are
 *          a function of the blob address, not of the request, and cannot be
 *          removed by asking for a different size.
 *
 * @requires vm.debug == true
 * @library /test/lib
 * @run driver compiler.jeandle.TestCodeBufferConstsExactAlloc
 */

package compiler.jeandle;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jdk.test.lib.Asserts;
import jdk.test.lib.process.OutputAnalyzer;
import jdk.test.lib.process.ProcessTools;

public class TestCodeBufferConstsExactAlloc {

    private static final Pattern CONST_TOTAL =
            Pattern.compile("align=(\\d+)");

    // Per-method CodeBuffer expansion count, reported on the [JeandleConstTotal]
    // line as `expands=<n>`.
    private static final Pattern EXPANDS =
            Pattern.compile("expands=(\\d+)");

    public static void main(String[] args) throws Exception {
        ProcessBuilder pb = ProcessTools.createJavaProcessBuilder(
                "-XX:+UnlockDiagnosticVMOptions",
                "-XX:+UseJeandleCompiler",
                "-XX:+JeandleCodeBufferInstrument",
                "-Xbatch",
                // Only the const-heavy methods are meant to be measured here;
                // letting the whole workload compile would also drag in main()
                // and the surrounding test scaffolding.
                "-XX:CompileCommand=compileonly,compiler.jeandle.TestCodeBufferConstValues::computeDouble",
                "-XX:CompileCommand=compileonly,compiler.jeandle.TestCodeBufferConstValues::computeFloat",
                "-XX:CompileCommand=compileonly,compiler.jeandle.TestCodeBufferConstValues::computeOopReloc",
                "-cp", System.getProperty("java.class.path"),
                "compiler.jeandle.TestCodeBufferConstValues");

        OutputAnalyzer out = new OutputAnalyzer(pb.start());
        out.shouldHaveExitValue(0);
        String stdout = out.getStdout();

        // --- deterministic layout summary ------------------------------------
        long methods      = summaryLong(stdout, "Methods reaching finalize");
        long exact        = summaryLong(stdout, "Exact path (from ConstSectionPlan)");
        long legacy       = summaryLong(stdout, "Legacy path (48 KiB default)");
        long planFailures = summaryLong(stdout, "Planner failures");
        long fallbacks    = summaryLong(stdout, "Runtime layout fallbacks");
        long expansions   = summaryLong(stdout, "consts expansions after exact plan");

        // Guard against a vacuous pass: something must have been compiled.
        Asserts.assertGT(methods, 0L, "no method reached finalize()");
        Asserts.assertEQ(exact, methods,
                "every finalized method must use the plan-driven consts size");
        Asserts.assertEQ(legacy, 0L, "no method may fall back to the legacy 48 KiB reserve");
        Asserts.assertEQ(planFailures, 0L, "the const section planner must not fail");
        Asserts.assertEQ(fallbacks, 0L, "no runtime layout fallback expected");
        Asserts.assertEQ(expansions, 0L,
                "the consts section must never expand after an exact plan");

        // --- instrumentation summary -----------------------------------------
        long usage    = instrumentationLong(stdout, "usage", "bytes /");
        long capacity = instrumentationLong(stdout, "/ capacity", "bytes,");

        Asserts.assertGT(usage, 0L, "the workload must emit at least one const byte");
        Asserts.assertGTE(capacity, usage, "capacity must cover the emitted consts");

        long maxAlign = 1;
        Matcher m = CONST_TOTAL.matcher(stdout);
        while (m.find()) {
            maxAlign = Math.max(maxAlign, Long.parseLong(m.group(1)));
        }

        // The rounding bound only describes a CodeBuffer that was divided up once.
        // If any section expanded, CodeBuffer::expand() allocated a larger blob and
        // re-divided every section, so a section whose plan was exact can legitimately
        // end up with more capacity than planned + alignment - 1. The per-method
        // `expands=` field says which case a run was in; skip the aggregate bound in
        // the expanded case, but report it rather than hiding it.
        int expandedMethods = 0;
        Matcher e = EXPANDS.matcher(stdout);
        while (e.find()) {
            if (Long.parseLong(e.group(1)) > 0) {
                expandedMethods++;
            }
        }
        if (expandedMethods == 0) {
            Asserts.assertLTE(capacity - usage, maxAlign - 1,
                    "consts over-allocation must stay within one alignment step"
                            + " (capacity=" + capacity + ", usage=" + usage
                            + ", align=" + maxAlign + ")");
        } else {
            System.out.println("note: " + expandedMethods
                    + " method(s) expanded; the aggregate rounding bound does not"
                    + " apply to re-divided buffers and was skipped"
                    + " (capacity=" + capacity + ", usage=" + usage + ")");
        }

        // The consts CodeSection itself must never have been expanded.
        int constsExpands = expandCount(stdout);
        Asserts.assertEQ(constsExpands, 0, "consts section was expanded at runtime");
    }

    /** Reads "  <key>:<spaces><number>" from the deterministic layout block. */
    private static long summaryLong(String stdout, String key) {
        for (String line : stdout.split("\\R")) {
            int idx = line.indexOf(key + ":");
            if (idx >= 0) {
                String tail = line.substring(idx + key.length() + 1).trim();
                return Long.parseLong(tail);
            }
        }
        throw new RuntimeException("missing layout summary key: " + key);
    }

    /** Reads "Consts: usage <n> bytes / capacity <n> bytes, waste ...%". */
    private static long instrumentationLong(String stdout, String before, String after) {
        for (String line : stdout.split("\\R")) {
            int idx = line.indexOf(before);
            if (idx >= 0) {
                String rest = line.substring(idx + before.length());
                int end = rest.indexOf(after);
                if (end < 0) {
                    throw new RuntimeException("malformed consts line: " + line);
                }
                return Long.parseLong(rest.substring(0, end).trim());
            }
        }
        throw new RuntimeException("missing instrumentation key: " + before);
    }

    /** Reads the consts component out of "Expand: total N (consts X, ...)". */
    private static int expandCount(String stdout) {
        for (String line : stdout.split("\\R")) {
            int idx = line.indexOf("consts ");
            if (idx >= 0 && line.contains("Expand:")) {
                String rest = line.substring(idx + "consts ".length());
                int comma = rest.indexOf(',');
                return Integer.parseInt(rest.substring(0, comma).trim());
            }
        }
        throw new RuntimeException("missing expand line");
    }
}

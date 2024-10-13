#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# This code is adapted from IDAES codebase:
# https://github.com/IDAES/idaes-pse/blob/0e05ed1410b83b5509efe68aa9e49026404e8db0/.pylint/idaes_reporters.py#L1
# IDAES is made available under the following license

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

# Standard libs
import time

# Installed libs
from pylint.message import Message
from pylint.reporters import BaseReporter, text


class DisplayProgress(BaseReporter):
    """
    Display analyzed modules with total elapsed time since beginning of run.
    """

    name = "progress"
    time_format = "07.3f"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = None

    @property
    def elapsed(self):
        """
        Returns elapsed time since the linting process began
        """
        f = time.perf_counter
        if self._start is None:
            self._start = f()
            return 0.0
        return f() - self._start

    def on_set_current_module(self, module, filepath):
        if filepath is None:
            return
        self.writeln(string=f"{self.elapsed:{self.time_format}} {filepath}")

    def _display(self, layout):
        pass


class DisplayOutput(text.TextReporter):
    """
    Controls how messages are displayed from Pylint. Two key updates are:
    1. Summary output is suppressed
    2. Messages are re-formatted
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._map_pylint = {
            "critical": "error",
            "warning": "warning",
            "info": "notice",
        }

    def write_message(self, msg: Message) -> None:
        msg_type = self._map_pylint.get(msg.category, "notice")
        msg_title = f"{msg.msg_id} ({msg.symbol})"
        line = (
            f"::{msg_type} file={msg.path},line={msg.line},"
            f"endLine={msg.end_line},title={msg_title}::{msg.msg}"
        )
        self.writeln(line)

    def display_reports(self, *args, **kwargs):
        # avoid displaying summary sections for this reporter
        pass


def register(linter):
    "This function needs to be defined for the plugin to be picked up by pylint"
    linter.register_reporter(DisplayProgress)
    linter.register_reporter(DisplayOutput)

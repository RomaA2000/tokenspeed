# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Dedicated request-config structure for logprob return.

TokenSpeed keeps ``SamplingParams`` focused on *how to sample*; logprob return
config is about *what to report*, so it lives here. It exposes
``logprobs`` / ``prompt_logprobs`` / ``logprob_token_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_LOGPROB_TOKEN_IDS = 128


@dataclass
class LogprobParams:
    # top-N logprobs per OUTPUT (sampled) token; None = off, 0 = chosen only,
    # N>0 = top-N + chosen. (-1 = full vocab is reserved but NOT yet supported;
    # verify() rejects it rather than silently returning a single token.)
    logprobs: int | None = None
    # top-N logprobs per PROMPT token; same value semantics as ``logprobs``.
    prompt_logprobs: int | None = None
    # logprobs for specific token ids, per position.
    logprob_token_ids: list[int] | None = None
    # detokenize tokens in returned logprobs (was ``return_text_in_logprobs``).
    return_text: bool = False

    def num_logprobs(self) -> int | None:
        """Per-output-token logprob count, or None if not requested.

        NOTE: ``0`` is a valid requested state (chosen token only), distinct
        from ``None`` (not requested). Callers MUST branch on ``is None``, not
        on truthiness of the return value.
        """
        if self.logprobs is not None:
            return self.logprobs
        if self.logprob_token_ids:
            return len(self.logprob_token_ids)
        return None

    def num_prompt_logprobs(self) -> int | None:
        """Per-prompt-token logprob count, or None if not requested."""
        return self.prompt_logprobs

    @property
    def requested(self) -> bool:
        return (
            self.logprobs is not None
            or self.prompt_logprobs is not None
            or bool(self.logprob_token_ids)
        )

    def verify(self, vocab_size: int, max_logprobs: int) -> None:
        for name, val in (
            ("logprobs", self.logprobs),
            ("prompt_logprobs", self.prompt_logprobs),
        ):
            if val is None:
                continue
            if val < -1:
                raise ValueError(f"{name} must be >= -1, got {val}.")
            if val == -1:
                # Full-vocab logprobs are not implemented yet (neither the
                # sampler/output path nor the prompt path materializes a
                # vocab-sized result). Reject loudly instead of silently
                # returning a single-token logprob dict.
                raise ValueError(
                    f"{name}=-1 (full-vocab logprobs) is not supported yet; "
                    f"use a non-negative count."
                )
            resolved = val
            if resolved > max_logprobs:
                raise ValueError(
                    f"{name}={resolved} exceeds max_logprobs={max_logprobs}."
                )
        if self.logprob_token_ids is not None:
            n = len(self.logprob_token_ids)
            if n > MAX_LOGPROB_TOKEN_IDS:
                raise ValueError(
                    f"logprob_token_ids length {n} exceeds " f"{MAX_LOGPROB_TOKEN_IDS}."
                )
            for tid in self.logprob_token_ids:
                if not 0 <= tid < vocab_size:
                    raise ValueError(
                        f"logprob_token_ids must be in [0, {vocab_size}), "
                        f"got {tid}."
                    )

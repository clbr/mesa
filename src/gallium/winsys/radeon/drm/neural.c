/*
 * Buffer scoring AI
 *
 * Copyright (C) 2014 Lauri Kasanen   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "neural.h"
#include <limits.h>

static float clampf(const float in, const float min, const float max) {
	return in > max ? max : in < min ? min : in;
}

static float smootherstep(float x) {
	x = clampf(x, 0, 1);

	return x*x*x*(x*(x*6 - 15) + 10);
}

uint32_t ai_calculate_score(const float inputs[INPUT_NEURONS],
				const struct network * const net) {

	uint32_t i, j;
	float input_results[INPUT_NEURONS], hidden_results[HIDDEN_NEURONS];

	/* Input layer */
	for (i = 0; i < INPUT_NEURONS; i++) {
		input_results[i] = smootherstep(net->input[i].weight * inputs[i] +
					net->input[i].bias);
	}

	/* Hidden layer */
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		float weighted_sum = net->hidden[i].bias;
		for (j = 0; j < INPUT_NEURONS; j++) {
			weighted_sum += net->hidden[i].weights[j] * input_results[j];
		}
		hidden_results[i] = smootherstep(weighted_sum);
	}

	/* Output layer */
	float weighted_sum = net->output.bias;
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		weighted_sum += net->output.weights[i] * hidden_results[i];
	}

	const float score = clampf(smootherstep(weighted_sum), 0, 0.999f);

	return score * UINT_MAX;
}

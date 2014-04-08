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

#ifndef NEURAL_H
#define NEURAL_H

#include <stdint.h>

#define INPUT_NEURONS 3
#define HIDDEN_NEURONS 8
#define NEURAL_VARS 47

struct input_neuron {
	float weight;
	float bias;
};

struct neuron {
	float weights[INPUT_NEURONS];
	float bias;
};

struct output_neuron {
	float weights[HIDDEN_NEURONS];
	float bias;
};

struct network {
	struct input_neuron input[INPUT_NEURONS];
	struct neuron hidden[HIDDEN_NEURONS];
	struct output_neuron output;
};

uint32_t ai_calculate_score(const float inputs[INPUT_NEURONS],
				const struct network * const net);

#endif

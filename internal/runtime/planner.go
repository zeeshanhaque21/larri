// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package runtime

import (
	"context"
	"go.sovrenix.com/larri/internal/core"
)

// SizingPlanner lets workflow engines size a graph of models instead of a
// single language model. The same plan drives selection and placed-host checks.
type SizingPlanner interface {
	SizingPlan(context.Context, core.ModelSpec) (core.SizingPlan, error)
}

// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// InstanceBill is what the provider says one rented instance is costing.
type InstanceBill struct {
	At                time.Time `json:"at"`
	HourlyUSD         float64   `json:"hourly_usd"` // compute plus rented disk, as billed
	DownGB            float64   `json:"down_gb"`    // inbound data billed so far
	UpGB              float64   `json:"up_gb"`
	DownUSDPerTB      float64   `json:"down_usd_tb"`
	UpUSDPerTB        float64   `json:"up_usd_tb"`
	DiskGB            float64   `json:"disk_gb"`
	StorageUSDGBMonth float64   `json:"storage_usd_gb_month"`
}

// SpendUSD is what an instance has cost after running for elapsed: the hourly
// rate for that time plus the data moved. The provider's data counters update
// in batches, so this can trail the truth by a few seconds of transfer.
func (b InstanceBill) SpendUSD(elapsed time.Duration) float64 {
	return b.HourlyUSD*elapsed.Hours() + b.DownGB*b.DownUSDPerTB/1000 + b.UpGB*b.UpUSDPerTB/1000
}

// Billing is the provider's own account of a run, the figures a result should
// be priced from. The journal LARRI keeps counts compute at the quoted rate and
// knows nothing of the disk or the data moved.
type Billing struct {
	CreditBeforeUSD float64 `json:"credit_before_usd"`
	CreditAfterUSD  float64 `json:"credit_after_usd"`
	// CreditSpentUSD is the account balance that went missing during the run.
	// It is ground truth for the run only if nothing else spent from the
	// account meanwhile, and it can lag by the provider's billing batch.
	CreditSpentUSD float64      `json:"credit_spent_usd"`
	AtReady        InstanceBill `json:"at_ready"`
	AtEnd          InstanceBill `json:"at_end"`
	// BootUSD is what the instance had cost when it first answered: the part
	// of the bill an API does not have.
	BootUSD float64 `json:"boot_usd"`
	// RunUSD is the instance's total cost from its own counters.
	RunUSD float64 `json:"run_usd"`
}

// Biller reads billing from a provider. Vast is the only implementation; the
// interface is what lets the runner be tested without an account.
type Biller interface {
	Credit(ctx context.Context) (float64, error)
	Instance(ctx context.Context, id string) (InstanceBill, error)
}

// VastBilling reads Vast.ai's account and instance endpoints.
type VastBilling struct {
	Key  string
	Base string // default https://console.vast.ai/api/v0
	HTTP *http.Client
}

func (v VastBilling) get(ctx context.Context, path string, out any) error {
	base := v.Base
	if base == "" {
		base = "https://console.vast.ai/api/v0"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+v.Key)
	c := v.HTTP
	if c == nil {
		c = &http.Client{Timeout: 20 * time.Second}
	}
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("stratabench: billing: %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(out)
}

// Credit returns the account's prepaid balance in dollars.
func (v VastBilling) Credit(ctx context.Context) (float64, error) {
	var u struct {
		Credit float64 `json:"credit"`
	}
	if err := v.get(ctx, "/users/current/", &u); err != nil {
		return 0, err
	}
	return u.Credit, nil
}

// Instance reads one instance's billing. The data counters arrive in kilobytes.
func (v VastBilling) Instance(ctx context.Context, id string) (InstanceBill, error) {
	var raw struct {
		Instances json.RawMessage `json:"instances"`
	}
	if err := v.get(ctx, "/instances/"+id+"/", &raw); err != nil {
		return InstanceBill{}, err
	}
	type wire struct {
		DPHTotal   float64 `json:"dph_total"`
		DownBilled float64 `json:"inet_down_billed"`
		UpBilled   float64 `json:"inet_up_billed"`
		DownTB     float64 `json:"internet_down_cost_per_tb"`
		UpTB       float64 `json:"internet_up_cost_per_tb"`
		Disk       float64 `json:"disk_space"`
		Storage    float64 `json:"storage_cost"`
	}
	var one wire
	if json.Unmarshal(raw.Instances, &one) != nil || one.DPHTotal == 0 {
		var many []wire
		if err := json.Unmarshal(raw.Instances, &many); err != nil || len(many) == 0 {
			return InstanceBill{}, fmt.Errorf("stratabench: billing: instance %s not found", id)
		}
		one = many[0]
	}
	return InstanceBill{
		At: time.Now().UTC(), HourlyUSD: one.DPHTotal,
		DownGB: one.DownBilled / 1e6, UpGB: one.UpBilled / 1e6,
		DownUSDPerTB: one.DownTB, UpUSDPerTB: one.UpTB,
		DiskGB: one.Disk, StorageUSDGBMonth: one.Storage,
	}, nil
}

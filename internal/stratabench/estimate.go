// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"fmt"
	"math"

	"go.sovrenix.com/larri/internal/core"
)

// Constants below that are estimates say so. They come from one live run, and
// the first thing a second run should do is replace them with measurements.
const (
	// hoursPerMonth converts a gigabyte-month storage price to an hourly one.
	// Vast's own arithmetic: 124 GB at $0.8667 per GB-month billed $0.1493 per
	// hour, which is 124 x 0.8667 / 720.
	hoursPerMonth = 720.0

	// linkEfficiency is the share of a host's listed link speed Strata's
	// download achieved: about 90 MB/s on a listed 768 Mbps (96 MB/s).
	linkEfficiency = 0.90

	// maxPlausibleMBps caps the download speed estimated for very fast links.
	// It is a guess, not a measurement: the one run was on a link slower than
	// Strata's downloader or the mirror's CDN may limit, so how fast it goes
	// on a 5 Gbps host is unknown.
	maxPlausibleMBps = 200.0

	// prepareMinutes is the time after the download and before the server
	// answers: file verification, the prepared pack, and the draft layer. One
	// run showed at least 16 minutes of it on a 22 GB card before it was
	// stopped; 20 is a round estimate, not a measurement.
	prepareMinutes = 20.0
)

// Estimate is the all-in cost of one benchmark on one offer, from the
// provider's published prices. It exists because the quoted hourly rate left
// out the rented disk and the data moved, and a run billed far more than its
// quote.
type Estimate struct {
	HourlyUSD    float64 // compute plus the rented disk, per hour
	DownloadGB   float64
	DownloadMin  float64
	BootMin      float64 // download plus preparation, until the first answer
	BandwidthUSD float64
	BootUSD      float64 // hourly cost until ready, plus bandwidth
	TotalUSD     float64 // boot plus the measurement itself
	Known        bool    // false when the offer reports no link speed, so nothing was estimated
}

// EstimateRun prices a benchmark of benchHours on an offer.
//
// A figure the provider did not report counts as zero cost rather than being
// invented, and Known says whether the link speed, which drives the biggest
// term, was available.
func EstimateRun(o core.Offer, p Plan, benchHours float64) Estimate {
	base := o.PriceBaseHr
	if base == 0 {
		base = o.PriceHr
	}
	e := Estimate{
		HourlyUSD:  base + float64(p.DiskGB)*o.StorageUSDPerGBMonth/hoursPerMonth,
		DownloadGB: float64(p.DownloadGB),
	}
	mbps := o.NetDownMbps / 8 * linkEfficiency
	mbps = math.Min(mbps, maxPlausibleMBps)
	if mbps > 0 {
		e.Known = true
		e.DownloadMin = e.DownloadGB * 1000 / mbps / 60
	}
	e.BootMin = e.DownloadMin + prepareMinutes
	e.BandwidthUSD = e.DownloadGB * o.NetDownUSDPerTB / 1000
	e.BootUSD = e.HourlyUSD*e.BootMin/60 + e.BandwidthUSD
	e.TotalUSD = e.BootUSD + e.HourlyUSD*benchHours
	return e
}

// String is a one-line summary for logs.
func (e Estimate) String() string {
	return fmt.Sprintf("$%.2f/hr all-in, ready in ~%.0f min, ~$%.2f total (bandwidth $%.2f)",
		e.HourlyUSD, e.BootMin, e.TotalUSD, e.BandwidthUSD)
}

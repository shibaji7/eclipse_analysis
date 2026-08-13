# Handoff Spec: MMS Satellite Location Schematic + Magnetotail Time Series for 2021 Antarctic Eclipse

**Target agent:** Codex
**Originating idea:** Discussion in the Eclipse AE / EPS Letters revision project (2026-07-30/31) about checking magnetospheric state (substorm/reconnection signatures) during the 4 December 2021 Antarctic eclipse using MMS data, to address GRL Reviewer #1's Major Comment #2 (is there evidence of an eclipse-triggered substorm?)
**Status:** draft idea, approximation pending validation — the user has observed "blips" in several MMS parameters between ~07:00-08:00 UT on 4 Dec 2021 but has NOT yet confirmed spacecraft location relative to the magnetotail, so physical interpretation of those blips is not yet established (see Section 10)

## 1. Objective

Produce two figures for the manuscript (destined for the new Figure 5, the time-series figure — final figure numbering still being finalized, see Section 10):

1. **A schematic showing MMS constellation location relative to Earth**, not to scale, at the time of interest (~07:00-08:00 UT, 4 Dec 2021), to establish whether the spacecraft were actually in the magnetotail (plasma sheet/lobe) versus dayside magnetopause, magnetosheath, or radiation belts at that time. This is a prerequisite check, not just a cosmetic figure — it determines whether the time-series blips below are even physically relevant to a magnetotail/substorm interpretation.
2. **A time-series figure** of relevant MMS parameters (magnetic field, plasma moments, and energetic particle flux if available) over the eclipse interval, aligned in time with the eclipse obscuration/totality window and, ideally, with the AL/SML geomagnetic index already being added elsewhere in the manuscript (see the related "Add AL index panel" task in the main project), to assess whether the observed blips coincide with a substorm-like dipolarization/injection signature.

## 2. Background / context

The manuscript under revision reports a case study of eclipse-driven magnetosphere-ionosphere coupling during the 4 Dec 2021 Antarctic eclipse. A GRL reviewer asked whether there is direct evidence of an eclipse-triggered substorm in this event (the manuscript currently only shows AE index, not AL, and speculates about a substorm without directly demonstrating one). The user has separately obtained MMS data for this interval and has "old code" using `pyspedas` that they intend to reuse/adapt — the coding agent should locate and inspect this existing code before writing anything new (see Section 7).

This is exploratory analysis, not yet a settled manuscript figure — the user has not yet confirmed the spacecraft were in a magnetotail-relevant location, so the coding agent's first deliverable (the location schematic) is what determines whether the second deliverable (time-series interpretation) is meaningful at all.

## 3. Inputs

| Name | Type/shape | Units | Source |
|---|---|---|---|
| `mms_probes` | list, e.g. `['mms1','mms2','mms3','mms4']` | — | User to confirm which probes have usable data — may not be all four |
| `time_range` | start/end UT | UT | 2021-12-04 06:00 to 09:00 UT (matches existing manuscript Figure 1/2/3 panels; widen if needed to give context before/after the 07:00-08:00 UT blip window) |
| `mec_position` | per-probe position (GSM or GSE), radial distance, MLT | Re, hours | MMS MEC (Magnetic Ephemeris and Coordinates) product, loaded via `pyspedas.mms.mec()` |
| `fgm_bfield` | per-probe B-field vector time series (Bx, By, Bz) | nT | MMS FGM (fluxgate magnetometer), via `pyspedas.mms.fgm()` |
| `fpi_moments` | per-probe ion/electron velocity, density, temperature | km/s, cm^-3, eV | MMS FPI (Fast Plasma Investigation), via `pyspedas.mms.fpi()` — burst mode if available for this interval, else survey mode |
| `eis_hpca_flux` (optional) | per-probe energetic particle flux by energy channel | keV, flux units | MMS EIS/HPCA, if the user's existing data/code includes it — include only if readily available, not a hard requirement |
| `eclipse_obscuration` | obscuration fraction time series | fraction (0-1) | Existing pyEclipse-derived obscuration data already used elsewhere in the manuscript pipeline |
| `existing_pyspedas_code` | user's prior scripts | — | User has existing code; location/path TBD, ask the user before writing new loading code from scratch |

## 4. Outputs

| Name | Type/shape | Units | Notes |
|---|---|---|---|
| `mms_location_schematic` | figure file | — | Earth-centered schematic (Earth as a simple circle/sphere at origin), NOT to scale, showing each MMS probe's approximate position/direction at the time of interest, with a clear "not to scale" label. Should make it visually obvious whether the constellation was tailward, duskward/dawnward, or near the dayside magnetopause. |
| `mms_timeseries_figure` | figure file | — | Stacked time-series panels (Bx/By/Bz, ion velocity components, density; energetic particle flux if available) over the eclipse interval, with eclipse totality/obscuration and (if available) AL/SML overlaid or in an aligned panel, styled per the manuscript's existing figure conventions (SCUBAS figure style) |
| `interpretation_notes` | short text/markdown summary | — | Plain-language summary of what the location schematic implies (tail vs. not) and whether the time-series blips align with a plausible dipolarization/injection signature or not — this is for the user's own assessment, not a manuscript-ready claim |

## 5. Function / module signatures

```python
def load_mms_mec(probes: list[str], time_range: tuple[str, str]) -> "dict[str, PositionDataset]":
    """Load MMS MEC position/MLT/radial-distance data per probe via pyspedas."""

def load_mms_fgm_fpi(probes: list[str], time_range: tuple[str, str], mode: str = "srvy") -> "dict[str, FieldsPlasmaDataset]":
    """Load MMS FGM B-field and FPI ion/electron moments per probe via pyspedas.
    mode: 'brst' (burst) if available for this interval, else 'srvy' (survey)."""

def plot_mms_location_schematic(
    positions: "dict[str, PositionDataset]",
    time_of_interest: str,
    output_path: str,
) -> None:
    """Render an Earth-centered, not-to-scale schematic of MMS probe positions
    at the given time, labeling GSM/GSE quadrant, approximate MLT, and
    whether each probe is plasma-sheet/lobe/magnetosheath/dayside per MEC region flags if available."""

def plot_mms_timeseries(
    fields_plasma: "dict[str, FieldsPlasmaDataset]",
    eclipse_obscuration: "ObscurationData",
    al_sml_index: "IndexData | None",
    time_range: tuple[str, str],
    output_path: str,
) -> None:
    """Render stacked time-series panels (B-field components, plasma velocity/density,
    optional energetic particle flux) with eclipse obscuration/totality and
    AL/SML overlays, matching existing manuscript figure style."""
```
(Illustrative only — the coding agent should adapt signatures to match whatever structure the user's existing pyspedas code already uses, rather than imposing a new interface from scratch.)

## 6. Algorithm / process description

1. **Locate and read the user's existing pyspedas-based code first.** Do not write new MMS-loading code before checking what already exists and reusing/adapting it.
2. Load MEC position data for the chosen probes over the time range; determine each probe's GSM/GSE position, radial distance, and MLT at the time(s) of interest (especially 07:00-08:00 UT).
3. Render the location schematic: Earth as a simple circle at the origin, MMS probe(s) plotted as points/markers at a schematic (not physically scaled) distance in the correct relative direction (dayside/nightside, dawn/dusk), clearly labeled "not to scale." Include a simple sun-direction reference (e.g., an arrow) so the reader can judge day/night side at a glance.
4. Load FGM and FPI data for the same probes/interval. Plot stacked time series: Bx, By, Bz; ion bulk velocity components (especially Vx, since a bursty bulk flow would show as an earthward Vx spike if the probe is tailward); density; and energetic particle flux from EIS/HPCA if the user's existing code already provides it.
5. Overlay or align eclipse obscuration/totality timing, and AL/SML index if available, on the same time axis as the MMS panels.
6. Do NOT attempt to declare a physical conclusion (e.g., "this confirms an eclipse-triggered substorm") in the code or figure captions — that determination belongs to the user/co-authors based on the figures produced. The deliverable is the visualization and a plain-language description of what's present, not a claim.

## 7. What NOT to change

- Do not modify `eps.tex` or any existing manuscript figure scripts (Figures 1-4) as part of this task.
- Do not assume all four MMS probes have usable data for this interval — confirm with the user which probes/data products they already have.
- Do not fabricate or assume burst-mode data availability — MMS burst mode is selectively triggered and may not cover this interval; check and fall back to survey mode if needed, and note which mode was used in the figure/caption.
- Do not make a substorm-onset determination on the coding agent's own authority — flag findings back to the user for the physical interpretation, per Section 6, step 6.

## 8. Acceptance criteria

- Location schematic clearly and correctly shows each probe's approximate quadrant (day/night, dawn/dusk) and rough radial distance category (e.g., inside ~10 Re vs. deep tail), verified by the coding agent cross-checking the plotted quadrant against the raw GSM/GSE coordinate signs (a simple, testable sanity check).
- Time-series figure time axis correctly spans and aligns with the 06:00-09:00 UT (or wider, if needed) window used elsewhere in the manuscript, with eclipse totality clearly marked.
- If AL/SML data is available, its dip timing (if any) is visually alignable against the MMS blip timing — this is the key comparison the user wants to make.
- Figures render without errors on the user's existing data files re-run end-to-end (not just on a single cached example).
- Include a short written note (in `interpretation_notes`) stating plainly whether the probes were tailward or not at the time in question — this is the single most important fact this task needs to establish before anything else about the blips can be claimed.

## 9. Source material

- User's existing pyspedas-based code (location to be provided by the user — ask if not immediately found in the working directory).
- MMS Science Data Center: https://lasp.colorado.edu/mms/sdc/public/ (data access, and daily quicklook plots useful for a fast manual sanity check before/alongside the coding task)
- `pyspedas` documentation for `mms_load_mec`, `mms_load_fgm`, `mms_load_fpi` routines.
- Related, already-in-progress project task: adding an AL/SML index panel to the manuscript's main time-series figure (see the "Add AL index panel to Fig 4 / address substorm question" task in the broader Eclipse AE project) — the AL/SML data source used there should be reused here for consistency, rather than pulled from a second, different source.
- Manuscript figure style: SCUBAS figure style conventions already used elsewhere in this project (color palette, font, layout) — apply the same conventions for visual consistency across all manuscript figures.

## 10. Open questions / known approximations

- **Not yet confirmed: were the MMS probes actually in the magnetotail at 07:00-08:00 UT on 4 Dec 2021?** This is the central open question this task exists to answer. Do not assume tail location going in.
- Final manuscript figure numbering is still being finalized in a separate, ongoing restructuring effort (the manuscript is being reorganized from a 4-figure to a 5-figure architecture) — treat "Figure 5" as a working label, not a fixed target; confirm final placement with the user before finalizing figure files for submission.
- Which MMS probes have usable/burst-mode data for this specific interval is unconfirmed — check data availability before assuming a specific probe set.
- Whether EIS/HPCA energetic particle data is readily available alongside the user's existing FGM/FPI code is unconfirmed.
- The AL/SML data source/format to use for the overlay should match whatever is chosen for the separate AL-index task elsewhere in the project — confirm this rather than picking a new source independently.
- The user mentioned rolling back an earlier plan to split Figure 1 into separate MCM/FIR figures back to a single combined Figure 1 — this is unrelated to the MMS task but is a live, unresolved decision elsewhere in the manuscript; the coding agent should not assume any particular figure-numbering scheme is final.
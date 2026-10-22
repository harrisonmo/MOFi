# Third-party notices

- Crystal-Builder fork: https://github.com/harrisonmo/Crystal-Builder
- Upstream: https://github.com/JulesOpp/Crystal-Builder
- Experiment baseline: df014447f0812dd4f55ed60359b71a981c3eeb60
- License: MIT; Copyright (c) 2026 Jules Oppenheim.

Crystal-Builder is a separate dependency, not bundled here. Preserve its LICENSE and third-party notices when obtaining or distributing it. Our Apache-2.0 license does not relicense that project or its materials. Benchmark records derived from sample structures retain attribution in experiment manifests. Other installed packages retain their own licenses.

## RASPA3 methane example

`examples/gcmc/MFI_SI.cif` is copied from iRASPA/RASPA3 commit
564b89329c1fd302b12c382995ad08d25d2c41ba,
`examples/basic/8_mc_adsorption_of_methane_in_mfi/MFI_SI.cif`.
Crystallographic attribution remains in the file. Parameters in
`examples/gcmc/mfi-force-field.json` are adapted from that example's
`force_field.json`, with neutral-site treatment and corrected Si element metadata.
RASPA3's MIT license and copyright are included in
`examples/gcmc/RASPA_LICENSE.md`. These inputs are an attributed MFI example,
not a newly developed MOF force field. Optional raspalib and ASE dependencies
retain their own licenses.

## ARIA (optional dependency)

The optional MCP reasoning connector uses
[yicao-elina/ARIA](https://github.com/yicao-elina/ARIA), pinned at
`cff9080f57360b1e89509adb3f298824eeed809d` in the `aria` extra of `pyproject.toml`.
ARIA is MIT-licensed, copyright 2026 ARIA Team. Its code and bundled knowledge graph
remain in the separately installed dependency; they are not vendored here. See the
[upstream license](https://github.com/yicao-elina/ARIA/blob/cff9080f57360b1e89509adb3f298824eeed809d/LICENSE).
The MCP Python SDK is also an optional MIT-licensed dependency. Demo request JSON
files in this repository are inputs for ARIA's bundled 2D-material example, not
new MOF scientific evidence.

# Refactoring Instructions: Modularizing `beamforming-export-scikit-maad.py`

This guide describes how to split the main script into maintainable modules. Each module should be placed in the same directory or a `beamforming_utils/` subdirectory. Update imports in your main script accordingly.

---

## 1. Direction Grid Helpers (`beamforming_grid.py`)
Move these functions:
- `latlong_grid`
- `fibonacci_sphere_grid`
- `unit_vector_from_azel`
- `angular_separation_deg`

**Purpose:**
Provides utilities for generating and working with direction grids and vectors.

[x] - Done 

---

## 2. Analysis Helpers (`beamforming_analysis.py`)
Move these functions:
- `calculate_correlation_analysis`
- `calculate_uniqueness_metrics`
- `smart_direction_selection`

**Purpose:**
Handles all ecoacoustic index calculations, correlation analysis, and direction selection logic.

[x] - Done 

---

## 3. Export Helpers (`beamforming_export.py`)
Move these functions:
- `export_selected_directions`
- `export_non_selected_directions`

**Purpose:**
Handles exporting mono WAVs for selected and non-selected directions.

[x] - Done 

---

## 4. Reporting & Visualization (`beamforming_report.py`)
Move these functions:
- `create_selection_report`
- `create_selection_visualization`
- `plot_exported_spectrograms`
- `generate_individual_spectrograms`
- `create_html_report`

**Purpose:**
Handles all reporting, visualization, and HTML report generation.

---

## 5. Main Script (`beamforming-export-scikit-maad.py`)
- Keep the workflow orchestration and configuration here.
- Import functions from the above modules.

---

## Example Imports
```python
from beamforming_grid import latlong_grid, fibonacci_sphere_grid, unit_vector_from_azel, angular_separation_deg
from beamforming_analysis import calculate_correlation_analysis, calculate_uniqueness_metrics, smart_direction_selection
from beamforming_export import export_selected_directions, export_non_selected_directions
from beamforming_report import create_selection_report, create_selection_visualization, plot_exported_spectrograms, generate_individual_spectrograms, create_html_report
```

---

## Benefits
- **Maintainability:** Each module has a clear responsibility.
- **Reusability:** Helpers can be reused in new scripts or workflows.
- **Clarity:** Easier for new contributors to understand and extend.

---

## After Refactoring
- Test each module independently.
- Update any relative imports if you use a subdirectory (e.g., `from .beamforming_grid import ...`).
- Keep configuration and workflow logic in the main script.

---

For questions, see the project README or ask the maintainers.

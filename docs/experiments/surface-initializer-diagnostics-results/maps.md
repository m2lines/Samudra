<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Salinity diagnostic map gallery

All panels use fixed salinity-difference limits, a 360 × 180 grid, and exactly 2 × 2 output pixels per cell. Open individual images at full size. See the [results](index.md) for metrics and interpretation.

![Common scale](figures/legend-2x.png)

## Shared training example: 1975-04-03

This is the **single example used to fit the one-example model**, and is also one of the sixteen examples used to fit the sixteen-example model (training index 0). Both models received 300 updates. These are fitting-set reconstructions of 550 m salinity, not held-out predictions. All images use the same color limits as the held-out gallery and exactly 2 × 2 pixels per grid cell.

| Model | RMSE on this example | Spatial anomaly correlation |
| --- | --- | --- |
| Original D | 0.03281 | 0.817 |
| Fitted to one example | 0.00642 | 0.994 |
| Fitted to sixteen examples | 0.01320 | 0.973 |

The sixteen-example numbers here describe **this one shared example**, not the average over its sixteen training examples.

### Outputs on the same example

| True anomaly | Original D anomaly |
| --- | --- |
| ![True anomaly](figures/fit-one-1975-04-03-2x-panel1.png) | ![Original D anomaly](figures/precision-1975-04-03-2x-panel2.png) |

| One-example model anomaly | Sixteen-example model anomaly |
| --- | --- |
| ![One-example model anomaly](figures/fit-one-1975-04-03-2x-panel2.png) | ![Sixteen-example model anomaly](figures/fit-sixteen-1975-04-03-2x-panel2.png) |

### Prediction minus truth

| One-example model error | Sixteen-example model error |
| --- | --- |
| ![One-example error](figures/fit-one-1975-04-03-2x-panel3.png) | ![Sixteen-example error](figures/fit-sixteen-1975-04-03-2x-panel3.png) |

| Original D error | Climatology error |
| --- | --- |
| ![Original D error](figures/precision-1975-04-03-2x-panel3.png) | ![Climatology error](figures/fit-one-1975-04-03-2x-panel4.png) |

## 2014-11-04

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2014-11-04-2x-panel1.png) | ![Prediction](figures/precision-2014-11-04-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2014-11-04-2x-panel3.png) | ![Climatology error](figures/precision-2014-11-04-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2014-11-04-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2014-11-04-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2014-11-04-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2014-11-04-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2014-11-04-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2014-11-04-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2014-11-04-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2014-11-04-2x-panel4.png) |

## 2015-02-02

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2015-02-02-2x-panel1.png) | ![Prediction](figures/precision-2015-02-02-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2015-02-02-2x-panel3.png) | ![Climatology error](figures/precision-2015-02-02-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2015-02-02-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2015-02-02-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2015-02-02-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2015-02-02-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2015-02-02-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2015-02-02-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2015-02-02-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2015-02-02-2x-panel4.png) |

## 2015-05-03

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2015-05-03-2x-panel1.png) | ![Prediction](figures/precision-2015-05-03-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2015-05-03-2x-panel3.png) | ![Climatology error](figures/precision-2015-05-03-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2015-05-03-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2015-05-03-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2015-05-03-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2015-05-03-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2015-05-03-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2015-05-03-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2015-05-03-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2015-05-03-2x-panel4.png) |

## 2015-08-01

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2015-08-01-2x-panel1.png) | ![Prediction](figures/precision-2015-08-01-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2015-08-01-2x-panel3.png) | ![Climatology error](figures/precision-2015-08-01-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2015-08-01-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2015-08-01-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2015-08-01-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2015-08-01-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2015-08-01-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2015-08-01-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2015-08-01-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2015-08-01-2x-panel4.png) |

## 2018-11-14

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2018-11-14-2x-panel1.png) | ![Prediction](figures/precision-2018-11-14-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2018-11-14-2x-panel3.png) | ![Climatology error](figures/precision-2018-11-14-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2018-11-14-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2018-11-14-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2018-11-14-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2018-11-14-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2018-11-14-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2018-11-14-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2018-11-14-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2018-11-14-2x-panel4.png) |

## 2022-11-24

### Original D

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/precision-2022-11-24-2x-panel1.png) | ![Prediction](figures/precision-2022-11-24-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/precision-2022-11-24-2x-panel3.png) | ![Climatology error](figures/precision-2022-11-24-2x-panel4.png) |

### Continued full-objective training

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/continued-control-eval-2022-11-24-2x-panel1.png) | ![Prediction](figures/continued-control-eval-2022-11-24-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/continued-control-eval-2022-11-24-2x-panel3.png) | ![Climatology error](figures/continued-control-eval-2022-11-24-2x-panel4.png) |

### Salinity-only fine-tuning

| True anomaly | Predicted anomaly |
| --- | --- |
| ![Truth](figures/field-specialization-eval-2022-11-24-2x-panel1.png) | ![Prediction](figures/field-specialization-eval-2022-11-24-2x-panel2.png) |

| Prediction minus truth | Climatology minus truth |
| --- | --- |
| ![Error](figures/field-specialization-eval-2022-11-24-2x-panel3.png) | ![Climatology error](figures/field-specialization-eval-2022-11-24-2x-panel4.png) |


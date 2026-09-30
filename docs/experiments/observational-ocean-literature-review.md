<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Literature review: surface-initialized ocean state prediction

Reviewed 2026-09-18. Companion to the [research plan](observational-ocean-prediction.md).
This is a targeted literature review, not an exhaustive systematic review or a reproduction of the results.
Preprints are identified below. Unless qualified, the paper's methods or full text were inspected;
performance claims are the authors' results.

The relevant task is surface-history initialization, followed by evolution under prescribed atmospheric forcing,
with future observed interior temperature and salinity as primary targets and daily SSH/SST as required surface
outputs. Forcing uses selected ERA5 atmospheric fields, not forecast-time OISST/DUACS. Model-data pretraining also
supervises interior velocities, which remain predicted during observational fine-tuning despite lacking gold
observational interior velocity targets. The literature supports the initialization and evolution components and
their joint training. Direct surface-to-future-interior prediction also has precedent, so the scientific contribution
should be measured improvement from simulation data and the resulting observational skill, rather than the existence
of a two-component architecture.

Three distinctions matter throughout: reconstructing the present versus forecasting the future; verification
against simulations/reanalyses versus actual profiles; and observing the surface versus supplying interior
information at initialization. The best atmospheric architectural analogues often have upper-air observations.

**Ocean: inferring an interior from the surface**

1. **Souza et al. (2025 preprint), Surface to Seafloor: A Generative AI Framework for Decoding the Ocean Interior State.**
   A likely match for the remembered paper. A conditional diffusion model infers three-dimensional velocity and
   temperature/buoyancy from sea surface height. It represents multiple plausible interiors and examines how
   depth and surface resolution affect uncertainty and dynamical statistics. The evidence is an idealized,
   15-level double-gyre primitive-equation simulation, not real observations. This is contemporaneous
   reconstruction, not future forecasting. It motivates a probabilistic initializer; it does not establish
   transfer to observed ocean profiles. [Paper](https://arxiv.org/abs/2504.15308).

2. **Miranda et al. (2025, Ocean Modelling), NeSPReSO.**
   An observational reconstruction baseline: satellite sea surface temperature, salinity, dynamic topography,
   location and season predict principal-component coefficients of temperature/salinity profiles to 1,800 m in
   the Gulf of Mexico. Training uses Argo, with comparisons against Argo and glider profiles. This suggests a
   relatively inexpensive profile initializer using a small network and a vertical basis. It does not learn
   future evolution; the exact split and observational overlap need auditing before copying its evaluation.
   [Paper](https://doi.org/10.1016/j.ocemod.2025.102550),
   [author-hosted full text](https://www.coaps.fsu.edu/pub/eric/papers_html/Miranda_et_al_25.pdf).

3. **Asefi et al. (2026 preprint), High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations.**
   Another possible remembered paper. Conditional diffusion reconstructs regional temperature, salinity and
   velocity from sparse surface height and temperature, with continuous depth conditioning. The Gulf of Mexico
   experiments use GLORYS for training and evaluation. This is a more realistic geometry than the double-gyre
   study, but agreement with GLORYS is not independent profile verification. It offers a depth-query decoder
   and a probabilistic initializer, rather than demonstrated future interior prediction.
   [Paper](https://arxiv.org/abs/2604.02850).

4. **Bolton and Zanna (2019, JAMES), Applications of Deep Learning to Ocean Data Inference and Subgrid Parameterization.**
   An earlier connection to this project's scientific context: CNNs learn subsurface inference and unresolved
   forcing from quasi-geostrophic simulations. Useful evidence that simulation-trained networks can extract
   surface/interior relationships, with substantial distance remaining to observed T/S forecasts.
   [Paper](https://doi.org/10.1029/2018MS001472),
   [author-hosted full text](https://zanna-research.github.io/files/Bolton-Zanna-2019.pdf).

**Ocean: predicting future interiors and transferring simulation knowledge**

5. **Liu et al. (2024, Ocean Modelling), Predicting temporal and spatial 4-D ocean temperature using satellite data based on a novel deep learning model.**
   Direct precedent for the forecast task: a month of surface history predicts two months of weekly subsurface
   temperature in the South China Sea. The model combines convolutions and attention. The authors compare with
   reanalysis and CTD measurements. A direct surface-history-to-future-interior network is therefore a relevant
   competitor to the proposed initializer plus dynamics model. Only the publisher abstract and public section
   excerpts were accessible: exact data ancestry, split boundaries and CTD independence remain unverified.
   [Paper](https://doi.org/10.1016/j.ocemod.2024.102333).

6. **Jiang et al. (2024, Remote Sensing), Multi-Scale Window Spatiotemporal Attention Network for Subsurface Temperature Prediction and Reconstruction (MSWO).**
   The closest direct monthly formulation found: 24 months of SST, salinity and dynamic topography predict six
   future months of temperature. Targets are monthly, 1-degree gridded Argo in the upper 250 m of the central
   Pacific; data span 2011–2019. The study separates forecasting from reconstruction with contemporaneous
   surface inputs. Its smoothed targets and short record limit comparison with individual future profiles.
   Sliding-window split boundaries would also need explicit checking before reproduction.
   [Paper](https://doi.org/10.3390/rs16122243),
   [full-text copy](https://pdfs.semanticscholar.org/295f/1c49ffdef5fb76f20472b4a55891a4ed7e1f.pdf).

7. **Ham, Kim and Luo (2019, Nature), Deep learning for multi-year ENSO forecasts.**
   A relevant simulation-transfer precedent at seasonal leads: pretraining on CMIP5 simulations followed by
   transfer to a reanalysis-based ENSO task. The target is an ENSO index rather than a global interior field.
   This supports trying simulation pretraining, without answering whether finer-resolution OM4/LLC improves
   profile-level skill. The review here uses the publisher abstract and authors' project materials.
   [Paper](https://doi.org/10.1038/s41586-019-1559-7),
   [author project](https://aiclimate.snu.ac.kr/enso/).

8. **Botvynko et al. (2025 preprint), Neural ocean forecasting from sparse satellite-derived observations: a case-study for SSH dynamics and altimetry data.**
   U-Net and 4DVarNet models learn seven-day surface forecasts from sparsely sampled histories, using GLORYS12
   with simulated satellite sampling for training and real altimetry for evaluation. Relevant to the transition
   from model-derived training inputs to actual observations, although the forecast target remains the surface.
   [Paper](https://arxiv.org/abs/2512.22152).

9. **Espinosa et al. (2026 preprint), DLESyM-Ocean.**
   A probabilistic ocean emulator with four-day steps, trained on an ERA5/UFS-Replay-derived dataset and driven
   by atmospheric fields. The authors report stable multidecadal integrations. It is relevant to the dynamics
   component and ensemble training, but initializes with a full ocean state and does not establish our
   surface-initialized, future-profile task. Stability and short-lead observational accuracy remain separate
   questions. [Paper](https://arxiv.org/abs/2608.11545).

**Ocean: reusable datasets and evaluation infrastructure**

10. **Donike et al. (2026 preprint), OceanDepths.**
    Provides paired satellite products, EN4 profiles and GLORYS, with raw and aligned data products. Potentially
    useful for preparation and matching. Its baseline supplies some contemporaneous interior profiles as inputs,
    which changes the task. Its 2018 evaluation split trains on other years through 2024, so it is not the
    chronological forecast split required here. Prefer profile-preserving products for our primary evaluation;
    dense gridded/quantized exports are not interchangeable with raw measurements.
    [Paper](https://arxiv.org/abs/2608.16373),
    [dataset](https://huggingface.co/datasets/ESA-philab/OceanDepths).

11. **El Aouni et al. (2025, NeurIPS Datasets and Benchmarks), OceanBench.**
    A global ocean forecasting benchmark with verification against reanalysis, operational analyses and
    observations, including CLASS4 profile comparisons. It also provides forecast trajectories between
    assimilation updates. Reuse the observation-matching/evaluation ideas while adapting initialization and
    lead times. The reviewed evidence is the conference abstract and official repository/documentation;
    this is not a full audit of its implementation. Do not confuse it with the earlier SSH-focused OceanBench.
    [Paper](https://papers.nips.cc/paper_files/paper/2025/hash/0d1635dfa43efe7477f9b6dab3168fd8-Abstract-Datasets_and_Benchmarks_Track.html),
    [official repository](https://github.com/mercator-ocean/oceanbench).

**Atmosphere: the closest architectural and training analogues**

12. **Allen et al. (2025, Nature), End-to-end data-driven weather prediction (Aardvark Weather).**
    The closest overall recipe: observations feed a state-estimation encoder, a dynamics processor evolves the
    state, and a decoder predicts station quantities. Components are pretrained using ERA5; the processor is
    then adapted to encoder-produced states, followed by joint observational fine-tuning. The reported joint
    fine-tuning experiment optimizes one-day station temperature/wind, although the system forecasts to ten
    days. Inputs include atmospheric sounders and radiosondes, so this is not surface-only interior inference.
    [Paper](https://doi.org/10.1038/s41586-025-08897-0),
    [author code and weights](https://github.com/anna-allen/aardvark-weather-public).

13. **Alexe et al. (2024 preprint), GraphDOP.**
    Heterogeneous observations are encoded into a latent representation, evolved and decoded into predictions
    at observation locations. Training is against observations rather than reanalysis targets. This supports
    learning dynamics and sparse observation losses without a dense observed state at every step. Inputs
    contain upper-air information; some quality control uses ERA5 departures. It is an alternative to requiring
    an explicitly physical intermediate state, not evidence that ocean surface observations uniquely determine
    the interior. [Paper](https://arxiv.org/abs/2412.15687).

14. **Xu et al. (2026 preprint), FuXiWeather2.**
    Separately pretrained atmospheric assimilation and forecast components are jointly trained through repeated
    cycles, reducing the mismatch between training backgrounds and the model's own outputs. Hybrid real and
    simulated observations help training. Here simulated observations largely mean observation-operator outputs,
    not simply additional free-running climate trajectories. This is relevant to training dynamics on inferred
    initial states and maintaining stability; an assimilation cycle receives new observations, whereas our
    forecast must exclude post-origin ocean observations. [Paper](https://arxiv.org/abs/2603.15358).

15. **Bodnar et al. (2025, Nature), A foundation model for the Earth system (Aurora).**
    Pretraining mixes large quantities of analyses, reanalyses, forecasts and simulations; fine-tuning adapts
    the model to different variables, resolutions and prediction problems. A useful precedent for treating
    heterogeneous data sources as an opportunity rather than insisting on one training dataset. Its forecasts
    start from gridded states. It does not directly establish the value of high-resolution ocean simulation
    data for observation-initialized interior prediction.
    [Paper](https://doi.org/10.1038/s41586-025-09005-y).

**Atmosphere: learning against observations and inferring unobserved state**

16. **Yuval et al., Neural general circulation models for modeling precipitation (Science Advances, 2026; preprint 2024).**
    NeuralGCM combines ERA5 state supervision with direct satellite precipitation targets. It demonstrates that
    a dynamical model need not inherit the reanalysis's errors in the observable being optimized. The study
    explicitly addresses inconsistent moisture budgets between the two target sources by changing losses and
    physical treatment. Relevant to combining OM4/GLORYS supervision with actual profiles: teacher agreement
    and observational accuracy can conflict. It uses full-state initialization.
    [Published-paper record](https://research.google/pubs/neural-general-circulation-models-for-modeling-precipitation/),
    [reviewed preprint](https://arxiv.org/abs/2412.11973).

17. **Schmitt et al. (September 2026 preprint), Improving precipitation forecasts in an AI weather model using observational data (Laxmi).**
    AIFS-CRPS is adapted to satellite IMERG precipitation targets while retaining reanalysis supervision for
    atmospheric state variables. The authors report improved held-out precipitation forecasts. This is a recent
    analogue of fine-tuning a model on the actual observational target, but it retains ERA5 initialization and
    uses a satellite retrieval product rather than error-free ground truth. It does not demonstrate joint
    observation-based initialization. [Paper](https://arxiv.org/abs/2609.03210).

18. **Slivinski et al. (2021, Journal of Climate), evaluation of Twentieth Century Reanalysis version 3 (20CRv3).**
    The closest classical boundary-observation analogy: a physical atmospheric model assimilates only surface
    pressure, with prescribed sea surface temperature and sea ice, to estimate the three-dimensional atmosphere
    and uncertainty. This shows why dynamics and observation history can constrain an unobserved interior.
    Continuous assimilation is a reconstruction task; it does not show that forecasting without further
    observations is equally well constrained, or that atmospheric observability transfers quantitatively to the
    ocean. [Paper](https://doi.org/10.1175/JCLI-D-20-0505.1),
    [NOAA documentation](https://psl.noaa.gov/data/20thC_Rean/).

**Implications for our experiments — synthesis, not claims made by a single paper**

- Keep the proposed initializer plus dynamics model as a strong candidate. Train dynamics on reconstructed
  initial states before joint fine-tuning, following the practical lesson from Aardvark and FuXiWeather2.
- Predict daily SSH/SST for the existing observational metrics, even if the interior evolves at coarser steps.
  Supply atmospheric forcing; reserve future OISST/DUACS for training labels and evaluation. Their pre-origin
  history remains a permitted initialization input. Match the time support of daily versus prepared products.
- Retain model-supervised interior velocity outputs during fine-tuning on surface and interior T/S observations.
  Mixing eligible model-output examples with velocity losses is an option to discourage forgetting. Assess
  plausibility separately: SSH-derived surface geostrophic metrics are not gold interior velocity observations.
- Include a direct surface-history-to-future-profile predictor as a serious competing approach. Existing ocean
  studies make it a natural baseline; the two-stage factorization should earn its complexity through skill.
  Use the same permitted atmospheric forcing and surface history for fair comparisons.
- Treat sparse profiles as labels at their measured coordinates, times and depths. A differentiable observation
  decoder can connect those labels to a full-state or latent dynamics model. Dense observed interior fields
  are not required. Monthly leads can use direct transitions; they do not require daily unrolling. This also
  does not convert monthly averages into instantaneous profile predictions.
- Allow probabilistic initialization. Compare ensemble means using point errors, and distributions using
  proper probabilistic scores and calibration. A visually realistic sample is not evidence that its eddies
  coincide with observed eddies.
- Distinguish paired data within a simulation from paired observations and reanalysis. Matching calendar dates
  does not make a free-running OM4 interior a label for the actual observed surface. Reanalysis can provide an
  observation-aligned teacher, with assimilation ancestry and temporal exclusions recorded.
- Keep real future profiles as the main reference. Argo-derived gridded products and GLORYS remain useful
  training sources and diagnostics, but measure a different problem. Prevent held-out profiles from returning
  through assimilating products, normalization or overlapping target windows. Distinguish the sampled-profile
  score from unsampled global-ocean accuracy.
- Report improvement beyond seasonality and persistent reconstructed anomalies. At monthly to seasonal leads,
  climatological structure can produce apparently good absolute-temperature scores. This is an evaluation
  requirement, not a demand for a rigid sequence of experiments before trying the strongest model.
- The literature reviewed does not settle whether OM4/LLC pretraining, finer simulation resolution or additional
  simulation families improve this exact future-profile task. Nor does it establish century-scale skill from
  monthly forecast results. Those remain empirical questions for the project.

A useful first reading set is Aardvark for the training procedure, Surface to Seafloor for uncertainty in
initialization, NeSPReSO for an observational reconstruction baseline, MSWO for direct monthly prediction,
and OceanBench for profile verification. OceanDepths is worth inspecting as preparation infrastructure, with
its initialization assumptions and split replaced for this task.

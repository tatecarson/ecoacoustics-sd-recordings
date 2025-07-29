#!/usr/bin/env Rscript

# Ecoacoustic Indices Analysis Script
# Calculates: Acoustic Entropy (H), NDSI, ACI, BI, ADI, AEI
# Robust to mixed return structures from soundecology; avoids crashes on per-index errors.

# ----------------------------- #
# Libraries
# ----------------------------- #
suppressPackageStartupMessages({
  library(tuneR)        # WAV I/O
  library(seewave)      # Entropy (H, Hf, Ht)
  library(soundecology) # NDSI, ACI, BI, ADI, AEI
  library(ggplot2)      # Plots
  library(dplyr)        # Data wrangling
  library(tidyr)        # Reshaping
})

# ----------------------------- #
# Helpers
# ----------------------------- #

# Ensure output directory exists
.ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# Extract a numeric value from a (named) list, trying candidate names first,
# then falling back to the first numeric element found.
.extract_numeric <- function(x, candidates = NULL) {
  if (is.null(x)) return(NA_real_)
  if (!is.null(candidates)) {
    for (nm in candidates) {
      if (!is.null(x[[nm]]) && is.numeric(x[[nm]]) && length(x[[nm]]) >= 1) {
        return(as.numeric(x[[nm]][1]))
      }
    }
  }
  # Fallback: first numeric element
  for (nm in names(x)) {
    val <- x[[nm]]
    if (is.numeric(val) && length(val) >= 1) return(as.numeric(val[1]))
  }
  # Last resort: try unlist and pick first numeric
  ux <- try(unlist(x, use.names = TRUE), silent = TRUE)
  if (!inherits(ux, "try-error")) {
    num_ix <- which(sapply(ux, is.numeric))
    if (length(num_ix) > 0) return(as.numeric(ux[num_ix[1]]))
  }
  NA_real_
}

# Safe z-score that returns 0s when variance is zero (or all NA)
.zscore <- function(v) {
  if (all(is.na(v))) return(rep(NA_real_, length(v)))
  m <- mean(v, na.rm = TRUE)
  s <- sd(v, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(0, length(v)))
  (v - m) / s
}

# Normalize to [0,1] with NA/constant protection
.norm01 <- function(v) {
  if (all(is.na(v))) return(rep(NA_real_, length(v)))
  rng <- range(v, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || diff(rng) == 0) return(rep(0, length(v)))
  (v - rng[1]) / (rng[2] - rng[1])
}

# ----------------------------- #
# Single-file analysis
# ----------------------------- #
analyze_wav_file <- function(filepath, min_freq = 200, max_freq = 12000) {
  cat(paste0("\n📊 Analyzing: ", basename(filepath), "\n"))
  cat(paste(rep("-", 50), collapse = ""), "\n")

  # Initialize result row
  results <- list(
    filename = basename(filepath),
    duration = NA_real_,
    sample_rate = NA_real_,
    acoustic_entropy_H = NA_real_,
    spectral_entropy_Hf = NA_real_,
    temporal_entropy_Ht = NA_real_,
    NDSI = NA_real_,
    anthrophony = NA_real_,
    biophony = NA_real_,
    ACI = NA_real_,
    BI = NA_real_,
    ADI = NA_real_,
    AEI = NA_real_
  )

  tryCatch({
    # Read audio
    wav_obj <- readWave(filepath)

    # Coerce to mono if stereo (average channels)
    if (wav_obj@stereo) {
      wav_obj <- mono(wav_obj, "both")
    }

    # File info
    duration <- length(wav_obj@left) / wav_obj@samp.rate
    cat(paste("Duration:", round(duration, 2), "seconds\n"))
    cat(paste("Sample rate:", wav_obj@samp.rate, "Hz\n"))
    results$duration <- duration
    results$sample_rate <- wav_obj@samp.rate

    # ----- Acoustic Entropy (H, Hf, Ht) -----
    tryCatch({
      cat("\n🔸 Calculating Acoustic Entropy (H)...\n")
      H_value <- H(wav_obj, f = wav_obj@samp.rate, wl = 512, envt = "hil")
      if (is.numeric(H_value) && is.finite(H_value)) {
        results$acoustic_entropy_H <- as.numeric(H_value)
        cat("  Total Entropy (H):", round(results$acoustic_entropy_H, 4), "\n")
      }

      spec_data <- meanspec(wav_obj, f = wav_obj@samp.rate, wl = 512, plot = FALSE)
      H_spectral <- sh(spec_data)
      if (is.numeric(H_spectral) && is.finite(H_spectral)) {
        results$spectral_entropy_Hf <- as.numeric(H_spectral)
        cat("  Spectral Entropy (Hf):", round(results$spectral_entropy_Hf, 4), "\n")
      }

      env_data <- env(wav_obj, f = wav_obj@samp.rate, plot = FALSE)
      H_temporal <- th(env_data)
      if (is.numeric(H_temporal) && is.finite(H_temporal)) {
        results$temporal_entropy_Ht <- as.numeric(H_temporal)
        cat("  Temporal Entropy (Ht):", round(results$temporal_entropy_Ht, 4), "\n")
      }
    }, error = function(e) {
      cat("❌ Error calculating Acoustic Entropy:", e$message, "\n")
    })

    # ----- NDSI -----
    tryCatch({
      cat("\n🔸 Calculating NDSI...\n")
      ndsi_result <- ndsi(wav_obj,
                          fft_w = 1024,
                          anthro_min = 200, anthro_max = 1500,
                          bio_min = 2000,  bio_max  = 11000)

      # Typical names in soundecology: ndsi_left, anthro_left, bio_left
      results$NDSI       <- .extract_numeric(ndsi_result, c("ndsi_left", "NDSI", "ndsi"))
      results$anthrophony <- .extract_numeric(ndsi_result, c("anthro_left", "anthrophony_left", "anthrophony"))
      results$biophony    <- .extract_numeric(ndsi_result, c("bio_left", "biophony_left", "biophony"))

      if (is.finite(results$NDSI))       cat("  NDSI:", round(results$NDSI, 4), "\n")
      if (is.finite(results$anthrophony)) cat("  Anthrophony:", round(results$anthrophony, 4), "\n")
      if (is.finite(results$biophony))    cat("  Biophony:", round(results$biophony, 4), "\n")
    }, error = function(e) {
      cat("❌ Error calculating NDSI:", e$message, "\n")
    })

    # ----- ACI -----
    tryCatch({
      cat("\n🔸 Calculating ACI...\n")
      aci_result <- acoustic_complexity(wav_obj,
                                        min_freq = min_freq,
                                        max_freq = max_freq,
                                        j = 5, fft_w = 512)

      results$ACI <- .extract_numeric(aci_result, c("AciTotAll_left", "AciTotAll", "aci_total", "aci"))
      if (is.finite(results$ACI)) cat("  ACI Total:", round(results$ACI, 2), "\n")
    }, error = function(e) {
      cat("❌ Error calculating ACI:", e$message, "\n")
    })

    # ----- Additional Indices: BI, ADI, AEI -----
    cat("\n🔸 Calculating additional indices...\n")

    # BI (pass file path)
    tryCatch({
      bi_result <- bioacoustic_index(file = filepath,
                                     min_freq = min_freq,
                                     max_freq = max_freq,
                                     fft_w    = 512)
      results$BI <- .extract_numeric(bi_result, c("left_area", "area_left", "Area", "area"))
      if (is.finite(results$BI)) cat("  Bioacoustic Index (BI):", round(results$BI, 2), "\n")
    }, error = function(e) {
      cat("❌ Error calculating BI:", e$message, "\n")
      results$BI <- NA_real_
    })

    # ADI (pass file path)
    tryCatch({
      adi_result <- acoustic_diversity(file = filepath,
                                       max_freq     = max_freq,
                                       db_threshold = -50,
                                       freq_step    = 1000)
      results$ADI <- .extract_numeric(adi_result, c("adi_left", "left_adi", "ADI", "adi"))
      if (is.finite(results$ADI)) cat("  Acoustic Diversity Index (ADI):", round(results$ADI, 4), "\n")
    }, error = function(e) {
      cat("❌ Error calculating ADI:", e$message, "\n")
      results$ADI <- NA_real_
    })

    # AEI (pass file path)
    tryCatch({
      aei_result <- acoustic_evenness(file = filepath,
                                      max_freq     = max_freq,
                                      db_threshold = -50,
                                      freq_step    = 1000)
      results$AEI <- .extract_numeric(aei_result, c("aei_left", "left_aei", "AEI", "aei"))
      if (is.finite(results$AEI)) cat("  Acoustic Evenness Index (AEI):", round(results$AEI, 4), "\n")
    }, error = function(e) {
      cat("❌ Error calculating AEI:", e$message, "\n")
      results$AEI <- NA_real_
    })

    return(results)

  }, error = function(e) {
    cat("❌ Error analyzing file:", e$message, "\n")
    return(results)
  })
}

# ----------------------------- #
# Visualization
# ----------------------------- #
create_visualizations <- function(results_df, output_dir) {
  cat("\n📊 Creating visualizations...\n")
  .ensure_dir(output_dir)

  # Require at least one non-NA value in the plotted columns
  plot_cols <- c("acoustic_entropy_H", "NDSI", "ACI", "BI", "ADI", "AEI")
  any_data <- any(sapply(results_df[plot_cols], function(v) any(is.finite(v))))
  if (!any_data) {
    cat("⚠️ Skipping plots: no finite values in indices.\n")
    return(invisible(NULL))
  }

  # 1) Faceted bars by index
  p1 <- results_df %>%
    select(direction, all_of(plot_cols)) %>%
    pivot_longer(cols = -direction, names_to = "Index", values_to = "Value") %>%
    ggplot(aes(x = direction, y = Value, fill = Index)) +
    geom_bar(stat = "identity", position = "dodge", na.rm = TRUE) +
    facet_wrap(~Index, scales = "free_y", ncol = 2) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Acoustic Indices by Direction", x = "Direction", y = "Index Value")

  tryCatch({
    ggsave(file.path(output_dir, "indices_by_direction.png"),
           p1, width = 12, height = 8, dpi = 300)
  }, error = function(e) cat("⚠️ Could not save indices_by_direction.png:", e$message, "\n"))

  # 2) Normalized (0–1) values for potential radar or comparative use (saved as CSV)
  results_norm <- results_df %>%
    select(direction, all_of(plot_cols)) %>%
    mutate(across(-direction, .norm01))
  tryCatch({
    write.csv(results_norm, file.path(output_dir, "indices_normalized_0to1.csv"), row.names = FALSE)
  }, error = function(e) cat("⚠️ Could not save indices_normalized_0to1.csv:", e$message, "\n"))

  # 3) Entropy components comparison
  if (all(c("spectral_entropy_Hf", "temporal_entropy_Ht") %in% names(results_df))) {
    p2 <- results_df %>%
      select(direction, acoustic_entropy_H, spectral_entropy_Hf, temporal_entropy_Ht) %>%
      pivot_longer(cols = -direction, names_to = "Entropy_Type", values_to = "Value") %>%
      ggplot(aes(x = direction, y = Value, fill = Entropy_Type)) +
      geom_bar(stat = "identity", position = "dodge", na.rm = TRUE) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Entropy Components by Direction", x = "Direction", y = "Entropy Value")

    tryCatch({
      ggsave(file.path(output_dir, "entropy_components.png"),
             p2, width = 10, height = 6, dpi = 300)
    }, error = function(e) cat("⚠️ Could not save entropy_components.png:", e$message, "\n"))
  }

  # 4) NDSI components (anthrophony vs biophony)
  if (all(c("anthrophony", "biophony") %in% names(results_df))) {
    p3 <- results_df %>%
      select(direction, anthrophony, biophony) %>%
      pivot_longer(cols = -direction, names_to = "Component", values_to = "Value") %>%
      ggplot(aes(x = direction, y = Value, fill = Component)) +
      geom_bar(stat = "identity", position = "stack", na.rm = TRUE) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "NDSI Components: Anthrophony vs Biophony", x = "Direction", y = "Relative Contribution")

    tryCatch({
      ggsave(file.path(output_dir, "ndsi_components.png"),
             p3, width = 10, height = 6, dpi = 300)
    }, error = function(e) cat("⚠️ Could not save ndsi_components.png:", e$message, "\n"))
  }

  cat("✅ Visualizations saved (where possible).\n")
}

# ----------------------------- #
# Main analysis (batch)
# ----------------------------- #
main <- function(input_dir = "ecoacoustic_analysis",
                 output_dir = NULL,
                 min_freq = 200,
                 max_freq = 12000) {

  cat("\n🎵 ECOACOUSTIC INDICES ANALYSIS\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")

  if (is.null(output_dir)) output_dir <- input_dir
  .ensure_dir(output_dir)

  wav_files <- list.files(input_dir, pattern = "^ecoacoustic_.*\\.wav$", full.names = TRUE)
  if (length(wav_files) == 0) {
    cat("❌ No ecoacoustic WAV files found in the specified directory!\n")
    return(invisible(NULL))
  }

  cat(paste("\nFound", length(wav_files), "files to analyze\n"))

  results_list <- vector("list", length(wav_files))
  for (i in seq_along(wav_files)) {
    results_list[[i]] <- analyze_wav_file(wav_files[i], min_freq, max_freq)
    # Ensure we always return a list with expected names
    if (is.null(results_list[[i]])) {
      results_list[[i]] <- as.list(setNames(rep(NA_real_, 12),
                                            c("duration","sample_rate","acoustic_entropy_H","spectral_entropy_Hf",
                                              "temporal_entropy_Ht","NDSI","anthrophony","biophony",
                                              "ACI","BI","ADI","AEI")))
      results_list[[i]]$filename <- basename(wav_files[i])
    }
    # Add direction from filename
    results_list[[i]]$direction <- gsub("ecoacoustic_(.*)_[0-9]+s\\.wav", "\\1", basename(wav_files[i]))
  }

  results_df <- bind_rows(results_list) %>%
    relocate(filename, direction)

  # Save raw results
  csv_file <- file.path(output_dir, "acoustic_indices_results.csv")
  write.csv(results_df, csv_file, row.names = FALSE)
  cat(paste("\n💾 Results saved to:", csv_file, "\n"))

  # ----- Summary Stats -----
  cat("\n📈 SUMMARY STATISTICS\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  indices <- c("acoustic_entropy_H", "NDSI", "ACI", "BI", "ADI", "AEI")
  for (idx in indices) {
    v <- results_df[[idx]]
    cat("\n", idx, ":\n", sep = "")
    cat("  Mean:", round(mean(v, na.rm = TRUE), 4), "\n")
    cat("  SD:",   round(sd(v,   na.rm = TRUE), 4), "\n")
    if (all(is.na(v))) {
      cat("  Range: NA - NA\n")
    } else {
      cat("  Range:", round(min(v, na.rm = TRUE), 4), "-", round(max(v, na.rm = TRUE), 4), "\n")
    }
  }

  # ----- Biodiversity Ranking -----
  cat("\n🏆 DIRECTION RANKINGS\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")

  required_cols <- c("acoustic_entropy_H", "NDSI", "ACI", "ADI", "AEI")
  missing_cols <- setdiff(required_cols, names(results_df))
  if (length(missing_cols) > 0) {
    cat("⚠️ Missing required columns for biodiversity score calculation:",
        paste(missing_cols, collapse = ", "), "\n")
    results_ranked <- results_df
  } else {
    valid_results <- results_df %>% filter(if_all(all_of(required_cols), ~ !is.na(.)))
    if (nrow(valid_results) > 0) {
      results_ranked <- valid_results %>%
        mutate(
          biodiversity_score =
            .zscore(acoustic_entropy_H) +
            .zscore(NDSI) +
            .zscore(ACI) +
            .zscore(ADI) -
            .zscore(AEI)
        ) %>%
        arrange(desc(biodiversity_score)) %>%
        select(direction, biodiversity_score, acoustic_entropy_H, NDSI, ACI, ADI, AEI)
      print(results_ranked)
    } else {
      cat("⚠️ No rows with complete data for biodiversity score calculation\n")
      results_ranked <- results_df %>%
        select(direction, acoustic_entropy_H, NDSI, ACI, ADI, AEI)
      print(results_ranked)
    }
  }

  # ----- Plots -----
  create_visualizations(results_df, output_dir)

  # ----- Interpretation Guide -----
  report_file <- file.path(output_dir, "acoustic_indices_interpretation.txt")
  sink(report_file)
  cat("ECOACOUSTIC INDICES INTERPRETATION GUIDE\n")
  cat(paste(rep("=", 50), collapse = ""), "\n\n")

  cat("INDEX DESCRIPTIONS:\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  cat("\n1. ACOUSTIC ENTROPY (H):\n")
  cat("   - Measures overall acoustic diversity; 0–1 scale (higher = more diverse)\n")
  cat("   - Combines spectral (Hf) and temporal (Ht) entropy\n")
  cat("\n2. NDSI (Normalized Difference Soundscape Index):\n")
  cat("   - Ratio of biophony vs. anthrophony; -1 to +1\n")
  cat("   - Positive = more biological sound; negative = more anthropogenic\n")
  cat("\n3. ACI (Acoustic Complexity Index):\n")
  cat("   - Variability in sound intensity; sensitive to bird choruses\n")
  cat("   - No fixed range; depends on recording length and parameters\n")
  cat("\n4. BI, ADI, AEI:\n")
  cat("   - BI: Area under bioacoustic band(s)\n")
  cat("   - ADI: Shannon diversity across frequency bins\n")
  cat("   - AEI: Evenness (Gini-like) across frequency bins\n")

  cat("\n\nYOUR RESULTS INTERPRETATION:\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  if (exists("results_ranked") && nrow(results_ranked) > 0 && "biodiversity_score" %in% names(results_ranked)) {
    best_dir <- results_ranked$direction[1]
    cat("\nMost biodiverse direction:", best_dir, "\n")
    cat("This direction shows the highest composite biodiversity score,\n")
    cat("indicating relatively richer biological acoustic activity.\n")
  } else {
    cat("\nA composite biodiversity ranking could not be computed (insufficient complete data).\n")
  }

  mean_ndsi <- mean(results_df$NDSI, na.rm = TRUE)
  if (is.finite(mean_ndsi)) {
    if (mean_ndsi > 0.5) {
      cat("\nHigh mean NDSI (>0.5) suggests dominant biological sounds.\n")
    } else if (mean_ndsi < -0.5) {
      cat("\nLow mean NDSI (<-0.5) indicates substantial anthropogenic noise.\n")
    }
  }

  mean_h <- mean(results_df$acoustic_entropy_H, na.rm = TRUE)
  if (is.finite(mean_h)) {
    if (mean_h > 0.8) {
      cat("\nHigh entropy (>0.8) suggests complex, diverse soundscapes.\n")
    } else if (mean_h < 0.5) {
      cat("\nLow entropy (<0.5) suggests simple/quiet or dominated soundscapes.\n")
    }
  }
  sink()

  cat(paste("\n📄 Interpretation guide saved to:", report_file, "\n"))
  cat("\n✅ ANALYSIS COMPLETE!\n")

  invisible(results_df)
}

# ----------------------------- #
# Entry point
# ----------------------------- #
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    main()
  } else {
    main(input_dir = args[1])
  }
}

# Example:
# results <- main(input_dir = "ecoacoustic_analysis", min_freq = 200, max_freq = 12000)

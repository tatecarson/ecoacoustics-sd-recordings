#!/usr/bin/env Rscript

# Ecoacoustic Indices Analysis Script
# Analyzes output files from the Python beamforming script
# Calculates: Acoustic Entropy (H), NDSI, and ACI

# Load required libraries
library(tuneR)        # For reading WAV files
library(seewave)      # For acoustic entropy calculation
library(soundecology) # For NDSI and ACI calculations
library(ggplot2)      # For visualization
library(dplyr)        # For data manipulation
library(tidyr)        # For data reshaping

# Function to analyze a single WAV file
analyze_wav_file <- function(filepath, min_freq = 200, max_freq = 12000) {
  
  cat(paste("\n📊 Analyzing:", basename(filepath), "\n"))
  cat(paste(rep("-", 50), collapse=""), "\n")
  
  # Initialize results list with all expected columns
  results <- list(
    filename = basename(filepath),
    duration = NA,
    sample_rate = NA,
    acoustic_entropy_H = NA,
    spectral_entropy_Hf = NA,
    temporal_entropy_Ht = NA,
    NDSI = NA,
    anthrophony = NA,
    biophony = NA,
    ACI = NA,
    BI = NA,
    ADI = NA,
    AEI = NA
  )
  
  # Read the WAV file
  tryCatch({
    wav_obj <- readWave(filepath)
    
    # Get file info
    duration <- length(wav_obj@left) / wav_obj@samp.rate
    cat(paste("Duration:", round(duration, 2), "seconds\n"))
    cat(paste("Sample rate:", wav_obj@samp.rate, "Hz\n"))
    
    results$duration <- duration
    results$sample_rate <- wav_obj@samp.rate
    
    # 1. ACOUSTIC ENTROPY (H) - Total entropy combining spectral and temporal
    tryCatch({
      cat("\n🔸 Calculating Acoustic Entropy (H)...\n")
      
      # Calculate total entropy using seewave
      H_value <- H(wav_obj, 
                   f = wav_obj@samp.rate,
                   wl = 512,  # Window length for FFT
                   envt = "hil")  # Use Hilbert transform for envelope
      
      results$acoustic_entropy_H <- NA # Initialize to NA
      if (is.numeric(H_value) && is.finite(H_value)) {
        results$acoustic_entropy_H <- H_value
        cat(paste("  Total Entropy (H):", round(H_value, 4), "\n"))
      }
      
      # Also calculate spectral and temporal components separately
      # Spectral entropy
      spec_data <- meanspec(wav_obj, f = wav_obj@samp.rate, wl = 512, plot = FALSE)
      H_spectral <- sh(spec_data)
      if (is.numeric(H_spectral) && is.finite(H_spectral)) {
        results$spectral_entropy_Hf <- H_spectral
        cat(paste("  Spectral Entropy (Hf):", round(H_spectral, 4), "\n"))
      }
      
      # Temporal entropy
      env_data <- env(wav_obj, f = wav_obj@samp.rate, plot = FALSE)
      H_temporal <- th(env_data)
      if (is.numeric(H_temporal) && is.finite(H_temporal)) {
        results$temporal_entropy_Ht <- H_temporal
        cat(paste("  Temporal Entropy (Ht):", round(H_temporal, 4), "\n"))
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating Acoustic Entropy:", e$message, "\n"))
    })
    
    # 2. NDSI - Normalized Difference Soundscape Index
    tryCatch({
      cat("\n🔸 Calculating NDSI...\n")
      
      # Calculate NDSI with ecologically relevant frequency bands
      ndsi_result <- ndsi(wav_obj,
                         fft_w = 1024,
                         anthro_min = 200,    # Anthropogenic sounds
                         anthro_max = 1500,   # (low frequencies)
                         bio_min = 2000,      # Biological sounds
                         bio_max = 11000)     # (higher frequencies)
      
      # Extract values safely - check structure first
      if (!is.null(ndsi_result)) {
        if ("ndsi_left" %in% names(ndsi_result) && is.numeric(ndsi_result$ndsi_left)) {
          results$NDSI <- as.numeric(ndsi_result$ndsi_left)
          cat(paste("  NDSI:", round(results$NDSI, 4), "\n"))
        }
        if ("anthro_left" %in% names(ndsi_result) && is.numeric(ndsi_result$anthro_left)) {
          results$anthrophony <- as.numeric(ndsi_result$anthro_left)
          cat(paste("  Anthrophony:", round(results$anthrophony, 4), "\n"))
        }
        if ("bio_left" %in% names(ndsi_result) && is.numeric(ndsi_result$bio_left)) {
          results$biophony <- as.numeric(ndsi_result$bio_left)
          cat(paste("  Biophony:", round(results$biophony, 4), "\n"))
        }
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating NDSI:", e$message, "\n"))
    })
    
    # 3. ACI - Acoustic Complexity Index
    tryCatch({
      cat("\n🔸 Calculating ACI...\n")
      
      # Calculate ACI
      aci_result <- acoustic_complexity(wav_obj,
                                      min_freq = min_freq,
                                      max_freq = max_freq,
                                      j = 5,  # Temporal step in seconds
                                      fft_w = 512)
      
      # Extract ACI value safely
      if (!is.null(aci_result) && "AciTotAll_left" %in% names(aci_result)) {
        aci_value <- as.numeric(aci_result$AciTotAll_left)
        if (is.numeric(aci_value) && is.finite(aci_value)) {
          results$ACI <- aci_value
          cat(paste("  ACI Total:", round(results$ACI, 2), "\n"))
        }
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating ACI:", e$message, "\n"))
    })
    
    # Additional indices for comprehensive analysis
    cat("\n🔸 Calculating additional indices...\n")
    
    # Bioacoustic Index (BI)
    tryCatch({
      bi_result <- bioacoustic_index(wav_obj,
                                    min_freq = min_freq,
                                    max_freq = max_freq,
                                    fft_w = 512)
      if (!is.null(bi_result) && "left_area" %in% names(bi_result)) {
        bi_value <- as.numeric(bi_result$left_area)
        if (is.numeric(bi_value) && is.finite(bi_value)) {
          results$BI <- bi_value
          cat(paste("  Bioacoustic Index (BI):", round(results$BI, 2), "\n"))
        }
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating BI:", e$message, "\n"))
    })
    
    # Acoustic Diversity Index (ADI)
    tryCatch({
      adi_result <- acoustic_diversity(wav_obj,
                                     max_freq = max_freq,
                                     db_threshold = -50,
                                     freq_step = 1000)
      if (!is.null(adi_result) && "adi_left" %in% names(adi_result)) {
        adi_value <- as.numeric(adi_result$adi_left)
        if (is.numeric(adi_value) && is.finite(adi_value)) {
          results$ADI <- adi_value
          cat(paste("  Acoustic Diversity Index (ADI):", round(results$ADI, 4), "\n"))
        }
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating ADI:", e$message, "\n"))
    })
    
    # Acoustic Evenness Index (AEI)
    tryCatch({
      aei_result <- acoustic_evenness(wav_obj,
                                    max_freq = max_freq,
                                    db_threshold = -50,
                                    freq_step = 1000)
      if (!is.null(aei_result) && "aei_left" %in% names(aei_result)) {
        aei_value <- as.numeric(aei_result$aei_left)
        if (is.numeric(aei_value) && is.finite(aei_value)) {
          results$AEI <- aei_value
          cat(paste("  Acoustic Evenness Index (AEI):", round(results$AEI, 4), "\n"))
        }
      }
    }, error = function(e) {
      cat(paste("❌ Error calculating AEI:", e$message, "\n"))
    })
    
    return(results)
    
  }, error = function(e) {
    cat(paste("❌ Error analyzing file:", e$message, "\n"))
    return(results)  # Return initialized results with NAs
  })
}

# Function to create comparative visualizations
create_visualizations <- function(results_df, output_dir) {
  
  cat("\n📊 Creating visualizations...\n")
  
  # 1. Bar plot of all indices by direction
  p1 <- results_df %>%
    select(direction, acoustic_entropy_H, NDSI, ACI, BI, ADI, AEI) %>%
    pivot_longer(cols = -direction, names_to = "Index", values_to = "Value") %>%
    ggplot(aes(x = direction, y = Value, fill = Index)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~Index, scales = "free_y", ncol = 2) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Acoustic Indices by Direction",
         x = "Direction",
         y = "Index Value") +
    scale_fill_brewer(palette = "Set2")
  
  ggsave(file.path(output_dir, "indices_by_direction.png"), 
         p1, width = 12, height = 8, dpi = 300)
  
  # 2. Radar plot preparation (normalized values)
  results_norm <- results_df %>%
    select(direction, acoustic_entropy_H, NDSI, ACI, BI, ADI, AEI) %>%
    mutate(across(-direction, ~(. - min(., na.rm = TRUE)) / 
                              (max(., na.rm = TRUE) - min(., na.rm = TRUE))))
  
  # 3. Entropy components comparison
  p2 <- results_df %>%
    select(direction, acoustic_entropy_H, spectral_entropy_Hf, temporal_entropy_Ht) %>%
    pivot_longer(cols = -direction, names_to = "Entropy_Type", values_to = "Value") %>%
    ggplot(aes(x = direction, y = Value, fill = Entropy_Type)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Entropy Components by Direction",
         x = "Direction",
         y = "Entropy Value",
         fill = "Entropy Type") +
    scale_fill_manual(values = c("acoustic_entropy_H" = "#1f77b4",
                                "spectral_entropy_Hf" = "#ff7f0e",
                                "temporal_entropy_Ht" = "#2ca02c"),
                     labels = c("Total (H)", "Spectral (Hf)", "Temporal (Ht)"))
  
  ggsave(file.path(output_dir, "entropy_components.png"), 
         p2, width = 10, height = 6, dpi = 300)
  
  # 4. NDSI components (anthrophony vs biophony)
  p3 <- results_df %>%
    select(direction, anthrophony, biophony) %>%
    pivot_longer(cols = -direction, names_to = "Component", values_to = "Value") %>%
    ggplot(aes(x = direction, y = Value, fill = Component)) +
    geom_bar(stat = "identity", position = "stack") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "NDSI Components: Anthrophony vs Biophony",
         x = "Direction",
         y = "Relative Contribution",
         fill = "Sound Type") +
    scale_fill_manual(values = c("anthrophony" = "#d62728",
                                "biophony" = "#2ca02c"))
  
  ggsave(file.path(output_dir, "ndsi_components.png"), 
         p3, width = 10, height = 6, dpi = 300)
  
  cat("✅ Visualizations saved!\n")
}

# Main analysis function
main <- function(input_dir = "ecoacoustic_analysis", 
                 output_dir = NULL,
                 min_freq = 200,
                 max_freq = 12000) {
  
  cat("\n🎵 ECOACOUSTIC INDICES ANALYSIS\n")
  cat(paste(rep("=", 50), collapse=""), "\n")
  
  # Set output directory
  if (is.null(output_dir)) {
    output_dir <- input_dir
  }
  
  # Get list of WAV files
  wav_files <- list.files(input_dir, pattern = "^ecoacoustic_.*\\.wav$", 
                         full.names = TRUE)
  
  if (length(wav_files) == 0) {
    cat("❌ No ecoacoustic WAV files found in the specified directory!\n")
    return()
  }
  
  cat(paste("\nFound", length(wav_files), "files to analyze\n"))
  
  # Analyze each file
  results_list <- list()
  
  for (i in seq_along(wav_files)) {
    file_results <- analyze_wav_file(wav_files[i], min_freq, max_freq)
    
    if (!is.null(file_results)) {
      # Extract direction from filename
      direction <- gsub("ecoacoustic_(.*)_[0-9]+s\\.wav", "\\1", 
                       basename(wav_files[i]))
      file_results$direction <- direction
      
      results_list[[i]] <- file_results
    }
  }
  
  # Combine results into dataframe
  results_df <- bind_rows(results_list)
  
  # Save results to CSV
  csv_file <- file.path(output_dir, "acoustic_indices_results.csv")
  write.csv(results_df, csv_file, row.names = FALSE)
  cat(paste("\n💾 Results saved to:", csv_file, "\n"))
  
  # Create summary statistics
  cat("\n📈 SUMMARY STATISTICS\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  
  # Print summary for each index
  indices <- c("acoustic_entropy_H", "NDSI", "ACI", "BI", "ADI", "AEI")
  
  for (idx in indices) {
    cat(paste("\n", idx, ":\n", sep=""))
    cat(paste("  Mean:", round(mean(results_df[[idx]], na.rm = TRUE), 4), "\n"))
    cat(paste("  SD:", round(sd(results_df[[idx]], na.rm = TRUE), 4), "\n"))
    cat(paste("  Range:", round(min(results_df[[idx]], na.rm = TRUE), 4), 
              "-", round(max(results_df[[idx]], na.rm = TRUE), 4), "\n"))
  }
  
  # Identify most biodiverse direction
  cat("\n🏆 DIRECTION RANKINGS\n")
  cat(paste(rep("-", 40), collapse=""), "\n")

  # Project convention: Only calculate biodiversity_score if all required columns exist and have valid data
  required_cols <- c("acoustic_entropy_H", "NDSI", "ACI", "ADI", "AEI")
  missing_cols <- setdiff(required_cols, names(results_df))
  if (length(missing_cols) > 0) {
    cat("⚠️ Missing required columns for biodiversity score calculation:", paste(missing_cols, collapse=", "), "\n")
    results_ranked <- results_df
  } else {
    # Filter for rows with all required indices present (no NA)
    valid_results <- results_df %>%
      filter(if_all(all_of(required_cols), ~!is.na(.)))
    if (nrow(valid_results) > 0) {
      # Calculate biodiversity score using scale()[,1] for each index
      results_ranked <- valid_results %>%
        mutate(
          biodiversity_score = scale(acoustic_entropy_H)[,1] +
                               scale(NDSI)[,1] +
                               scale(ACI)[,1] +
                               scale(ADI)[,1] -
                               scale(AEI)[,1]
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

  # Create visualizations
  create_visualizations(results_df, output_dir)
  
  # Generate interpretation report
  report_file <- file.path(output_dir, "acoustic_indices_interpretation.txt")
  
  sink(report_file)
  cat("ECOACOUSTIC INDICES INTERPRETATION GUIDE\n")
  cat(paste(rep("=", 50), collapse=""), "\n\n")
  
  cat("INDEX DESCRIPTIONS:\n")
  cat(paste(rep("-", 30), collapse=""), "\n")
  
  cat("\n1. ACOUSTIC ENTROPY (H):\n")
  cat("   - Measures overall acoustic diversity\n")
  cat("   - Range: 0-1 (higher = more diverse)\n")
  cat("   - Combines spectral (Hf) and temporal (Ht) entropy\n")
  cat("   - High values indicate rich, varied soundscapes\n")
  
  cat("\n2. NDSI (Normalized Difference Soundscape Index):\n")
  cat("   - Ratio of biophony to anthrophony\n")
  cat("   - Range: -1 to +1\n")
  cat("   - Positive values = more biological sounds\n")
  cat("   - Negative values = more human-made sounds\n")
  
  cat("\n3. ACI (Acoustic Complexity Index):\n")
  cat("   - Measures variability in sound intensity\n")
  cat("   - Higher values indicate more biotic activity\n")
  cat("   - Particularly sensitive to bird vocalizations\n")
  cat("   - No fixed range (depends on recording length)\n")
  
  cat("\n4. Additional Indices:\n")
  cat("   - BI: Area under the curve in specific frequency bands\n")
  cat("   - ADI: Shannon diversity of frequency bands\n")
  cat("   - AEI: Gini coefficient of frequency bands (evenness)\n")
  
  cat("\n\nYOUR RESULTS INTERPRETATION:\n")
  cat(paste(rep("=", 50), collapse=""), "\n")
  
  best_dir <- results_ranked$direction[1]
  cat(paste("\nMost biodiverse direction:", best_dir, "\n"))
  cat("This direction shows the highest composite biodiversity score,\n")
  cat("indicating rich biological acoustic activity.\n")
  
  # Specific interpretations based on values
  mean_ndsi <- mean(results_df$NDSI, na.rm = TRUE)
  if (mean_ndsi > 0.5) {
    cat("\nHigh NDSI values (>0.5) indicate dominant biological sounds.\n")
    cat("This suggests a healthy ecosystem with active wildlife.\n")
  } else if (mean_ndsi < -0.5) {
    cat("\nLow NDSI values (<-0.5) indicate significant anthropogenic noise.\n")
    cat("Consider the impact of human activities on the soundscape.\n")
  }
  
  mean_h <- mean(results_df$acoustic_entropy_H, na.rm = TRUE)
  if (mean_h > 0.8) {
    cat("\nHigh entropy values (>0.8) suggest complex, diverse soundscapes.\n")
    cat("This typically indicates high species richness.\n")
  } else if (mean_h < 0.5) {
    cat("\nLow entropy values (<0.5) suggest simple or quiet soundscapes.\n")
    cat("This might indicate low biodiversity or specific dominant sounds.\n")
  }
  
  sink()
  
  cat(paste("\n📄 Interpretation guide saved to:", report_file, "\n"))
  cat("\n✅ ANALYSIS COMPLETE!\n")
  
  return(results_df)
}

# Run the analysis if script is executed directly
if (!interactive()) {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) == 0) {
    # Use default directory
    results <- main()
  } else {
    # Use specified directory
    results <- main(input_dir = args[1])
  }
}

# Example usage:
# results <- main(input_dir = "ecoacoustic_analysis", min_freq = 200, max_freq = 12000)
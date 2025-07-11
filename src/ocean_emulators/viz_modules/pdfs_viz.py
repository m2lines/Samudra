"""PDFs visualization module for ocean data."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Dict, List
from matplotlib.ticker import MaxNLocator


def create_pdfs_directory(output_path: str) -> str:
    """
    Create PDFs output directory.
    
    Args:
        output_path: Base output path
        
    Returns:
        Path to PDFs directory
    """
    pdfs_path = os.path.join(output_path, "PDFs")
    os.makedirs(pdfs_path, exist_ok=True)
    return pdfs_path


def compute_pdf(data: xr.DataArray, bins: int = 150, 
               data_range: tuple = None) -> tuple:
    """
    Compute probability density function for data.
    
    Args:
        data: Input data array
        bins: Number of bins for histogram
        data_range: (min, max) range for binning
        
    Returns:
        Tuple of (pdf_values, bin_edges)
    """
    # Flatten data and remove NaN values
    flat_data = data.values.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]
    
    # Use provided range or compute from data
    if data_range is None:
        data_range = (flat_data.min(), flat_data.max())
    
    # Compute histogram with density normalization
    pdf_values, bin_edges = np.histogram(
        flat_data, bins=bins, density=True, range=data_range
    )
    
    return pdf_values, bin_edges


def create_pdf_plots_short(ds_groundtruth: xr.Dataset,
                          pred_dict: Dict[str, Dict],
                          output_path: str,
                          titles: List[str],
                          dataset_name: str = "OM4") -> None:
    """
    Create PDF_Plots_Short.png - probability density function plots for key variables.
    
    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    print("📊 Starting PDF plots creation...")
    
    try:
        pdfs_path = create_pdfs_directory(output_path)
        
        # Color list for different models
        clist = ["#ff807a", "#1e8685", "#ffb579", "#63c8ab"]
        
        # Set plotting parameters
        plt.rcParams.update({"font.size": 9})
        
        print("🎨 Creating PDF figure layout...")
        # Create a figure with manual subplot positioning
        fig = plt.figure(figsize=(24, 15))
        plt.rc("axes", titlesize=30)  # fontsize of the axes title
        plt.rc("axes", labelsize=30)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=30)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=30)  # fontsize of the tick labels
        plt.rc("legend", fontsize=20)  # legend fontsize
        plt.rc("figure", titlesize=30)
        
        # Manual positioning using add_axes with uniform width and height
        width = 0.22
        height = 0.3
        
        # Top row: 3 plots, evenly spaced horizontally
        axs = [
            fig.add_axes([0.05, 0.55, width, height]),  # First plot in top row
            fig.add_axes([0.38, 0.55, width, height]),  # Second plot in top row
            fig.add_axes([0.71, 0.55, width, height]),  # Third plot in top row
        ]
        
        # Bottom row: 2 plots centered, manually positioned
        axs += [
            fig.add_axes([0.22, 0.1, width, height]),  # First plot in bottom row
            fig.add_axes([0.54, 0.1, width, height]),  # Second plot in bottom row
        ]
        
        # Variables to analyze
        variables = ["thetao", "so", "zos", "uo", "vo"]
        
        print(f"📈 Processing {len(variables)} variables for PDF analysis...")
        
        # Plot PDFs for each variable
        for i, v in enumerate(variables):
            print(f"🔍 Processing variable {i+1}/{len(variables)}: {v}")
            
            if v not in ds_groundtruth.data_vars:
                print(f"⚠️ Variable {v} not found in ground truth dataset, skipping...")
                # Create placeholder plot
                axs[i].text(0.5, 0.5, f"Variable {v}\nnot available", 
                           ha='center', va='center', transform=axs[i].transAxes, fontsize=16)
                axs[i].set_title(f"{v} (N/A)")
                continue
            
            try:
                print(f"📊 Computing data range for {v}...")
                # Calculate data range for consistent binning - use smaller sample for speed
                gt_sample = ds_groundtruth[v].isel(time=slice(0, min(10, len(ds_groundtruth.time))))
                min_val = float(gt_sample.min().values)
                max_val = float(gt_sample.max().values)
                
                # Include prediction data in range calculation
                for k, pred_data in pred_dict.items():
                    if v in pred_data["ds_prediction"].data_vars:
                        pred_sample = pred_data["ds_prediction"][v].isel(time=slice(0, min(10, len(pred_data["ds_prediction"].time))))
                        pred_min = float(pred_sample.min().values)
                        pred_max = float(pred_sample.max().values)
                        min_val = min(min_val, pred_min)
                        max_val = max(max_val, pred_max)
                
                data_range = (min_val, max_val)
                print(f"📈 Data range for {v}: {data_range}")
                
                # Compute ground truth PDF with smaller sample
                print(f"🧮 Computing ground truth PDF for {v}...")
                true_pdf, bins_true = compute_pdf(gt_sample, bins=100, data_range=data_range)
                
                # Plot ground truth PDF (log scale)
                axs[i].semilogy(bins_true[:-1], true_pdf, label=dataset_name, color="k", lw=8)
                
                # Plot prediction PDFs
                for j, (k, pred_data) in enumerate(pred_dict.items()):
                    if v in pred_data["ds_prediction"].data_vars:
                        print(f"🧮 Computing prediction PDF for {v} from {k}...")
                        # Compute prediction PDF with smaller sample
                        pred_sample = pred_data["ds_prediction"][v].isel(time=slice(0, min(10, len(pred_data["ds_prediction"].time))))
                        pdf_net, bins_net = compute_pdf(pred_sample, bins=100, data_range=data_range)
                        
                        # Get prediction name
                        pred_name = titles[j] if j < len(titles) else pred_data.get("name", k)
                        
                        # Plot prediction PDF (log scale)
                        axs[i].semilogy(
                            bins_net[:-1], pdf_net, label=pred_name, color=clist[j % len(clist)], lw=2
                        )
                
                # Format the subplot
                axs[i].xaxis.set_major_locator(MaxNLocator(5, prune="both"))
                if i == 0:
                    axs[i].legend()
                
                # Set axis labels using variable attributes
                var_data = ds_groundtruth[v]
                long_name = getattr(var_data, 'long_name', v)
                units = getattr(var_data, 'units', '')
                
                axs[i].set_xlabel(f"{long_name} [{units}]")
                axs[i].set_ylabel(f"$p({long_name})$")
                
                # Set y-axis limits based on variable type  
                if v not in ["thetao", "zos"]:  # For salinity, velocity variables
                    axs[i].set_ylim([1e-5, true_pdf.max() * 2])
                else:  # For temperature and SSH
                    axs[i].set_ylim([1e-3, true_pdf.max() * 2])
                
                print(f"✅ Completed PDF for {v}")
                
            except Exception as e:
                print(f"❌ Error processing {v}: {e}")
                # Create error placeholder
                axs[i].text(0.5, 0.5, f"Error processing\n{v}\n{str(e)[:50]}...", 
                           ha='center', va='center', transform=axs[i].transAxes, fontsize=12)
                axs[i].set_title(f"{v} (Error)")
        
        print("💾 Saving PDF plots...")
        # Reset matplotlib style
        matplotlib.style.use("default")
        
        # Save the figure
        plt.savefig(os.path.join(pdfs_path, "PDF_Plots_Short.png"), bbox_inches="tight", dpi=600)
        plt.close()
        
        print(f"✅ PDF plots saved to: {pdfs_path}/PDF_Plots_Short.png")
        
    except Exception as e:
        print(f"❌ Critical error in PDF plots creation: {e}")
        # Create minimal fallback plot
        try:
            pdfs_path = create_pdfs_directory(output_path)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f"PDF Analysis\nData processing error\n{str(e)[:100]}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title("PDF_Plots_Short (Error)")
            plt.savefig(os.path.join(pdfs_path, "PDF_Plots_Short.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"⚠️ Fallback PDF plot saved to: {pdfs_path}/PDF_Plots_Short.png")
        except:
            print("❌ Failed to create even fallback PDF plot")
            return


def generate_all_pdf_plots(ds_groundtruth: xr.Dataset,
                          pred_dict: Dict[str, Dict],
                          output_path: str,
                          titles: List[str],
                          dataset_name: str = "OM4") -> None:
    """
    Generate all PDF visualization plots.
    
    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Base output path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    print("Generating PDF plots...")
    
    # Create the main PDF plot
    create_pdf_plots_short(ds_groundtruth, pred_dict, output_path, titles, dataset_name)
    
    print("PDF plots completed!")
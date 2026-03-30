#!/usr/bin/env python3
"""
LiDAR Metrics Visualizer - Handles both standard CSV and concatenated formats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os
from pathlib import Path
from datetime import datetime
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_ros_csv(filepath):
    """
    Parse ROS CSV file - handles both standard CSV and concatenated formats.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠ Error reading {filepath}: {e}")
        return pd.DataFrame()
    
    if not content.strip():
        print(f"⚠ Empty file: {filepath}")
        return pd.DataFrame()
    
    # Try standard CSV format first
    try:
        # Check if it has proper newlines and looks like standard CSV
        lines = content.strip().split('\n')
        if len(lines) > 1 and ',' in lines[0]:
            # Try parsing as standard CSV
            df = pd.read_csv(filepath)
            
            # Check if we got valid data
            if not df.empty and 'stamp_sec' in df.columns:
                print(f"   ✓ Parsed as standard CSV: {len(df):,} records")
                
                # Extract metadata from topic names
                df[['config_type', 'sensor', 'metric_name']] = df['topic'].str.extract(
                    r'/metrics/(?P<config_type>\w+)/(?P<sensor>\w+)/(?P<metric_name>\w+)'
                )
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['stamp_sec'], unit='s')
                
                # Rename stamp_sec to timestamp for consistency
                df = df.rename(columns={'stamp_sec': 'timestamp'})
                
                return df
    except Exception as e:
        print(f"   → Standard CSV parsing failed, trying concatenated format...")
        pass
    
    # Fall back to regex-based parser for concatenated format
    pattern = r'(\d+\.\d+),(/metrics/[^,]+),(std_msgs/msg/\w+),(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, content)
    
    records = []
    for match in matches:
        try:
            record = {
                'timestamp': float(match[0]),
                'topic': match[1],
                'msg_type': match[2],
                'value': float(match[3])
            }
            records.append(record)
        except (ValueError, IndexError):
            continue
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        print(f"   ✓ Parsed as concatenated format: {len(df):,} records")
        
        # Extract metadata from topic names
        df[['config_type', 'sensor', 'metric_name']] = df['topic'].str.extract(
            r'/metrics/(?P<config_type>\w+)/(?P<sensor>\w+)/(?P<metric_name>\w+)'
        )
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    return df


def load_all_files(data_dir, pattern="metrics_*.csv"):
    """Load and combine all matching CSV files."""
    all_data = []
    
    search_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    print(f"📁 Searching: {search_pattern}")
    print(f"   Files found: {len(files)}")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\n   📄 {filename}")
        
        df = parse_ros_csv(filepath)
        
        if df.empty:
            print(f"      ⚠ No records parsed")
            continue
        
        # Extract config info from filename
        if 'dbscan' in filename.lower():
            df['config'] = 'DBSCAN'
        elif 'pcl' in filename.lower():
            df['config'] = 'PCL'
        else:
            df['config'] = 'Unknown'
            
        if 'fused' in filename.lower():
            df['mode'] = 'Fused'
        elif 'per_sensor' in filename.lower():
            df['mode'] = 'Per-Sensor'
        else:
            df['mode'] = 'Single'
            
        df['source_file'] = Path(filename).stem
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def plot_processing_time_comparison(df, output_dir="plots"):
    """Compare processing times across configurations and sensors."""
    Path(output_dir).mkdir(exist_ok=True)
    
    proc_df = df[df['metric_name'] == 'processing_ms'].copy()
    
    if proc_df.empty:
        print("\n⚠ No processing_ms data found")
        return
    
    sensors = proc_df['sensor'].dropna().unique()
    print(f"\n📊 Creating processing time plots for sensors: {list(sensors)}")
    
    for sensor in sensors:
        if pd.isna(sensor):
            continue
            
        plt.figure(figsize=(14, 5))
        
        for config in proc_df['config'].unique():
            subset = proc_df[
                (proc_df['sensor'] == sensor) & 
                (proc_df['config'] == config)
            ]
            if not subset.empty:
                plt.plot(
                    subset['datetime'], 
                    subset['value'],
                    label=f"{config}",
                    marker='.',
                    markersize=3,
                    alpha=0.7,
                    linewidth=1
                )
        
        plt.title(f'Processing Time - {str(sensor).capitalize()} Sensor')
        plt.xlabel('Time')
        plt.ylabel('Processing Time (ms)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/proc_time_{sensor}.png", dpi=150)
        plt.close()
        print(f"   ✓ Saved: proc_time_{sensor}.png")
    
    # Box plot comparison
    valid_df = proc_df.dropna(subset=['sensor', 'config'])
    if not valid_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=valid_df,
            x='sensor',
            y='value',
            hue='config',
            palette='Set2'
        )
        plt.title('Processing Time Distribution by Sensor & Configuration')
        plt.ylabel('Processing Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/proc_time_boxplot.png", dpi=150)
        plt.close()
        print(f"   ✓ Saved: proc_time_boxplot.png")


def plot_detections_comparison(df, output_dir="plots"):
    """Compare detection counts across configurations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    det_df = df[df['metric_name'] == 'detections_count'].copy()
    
    if det_df.empty:
        print("⚠ No detections_count data found")
        return
    
    summary = det_df.groupby(['config', 'sensor'])['value'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary.dropna(subset=['sensor']),
        x='sensor',
        y='mean',
        hue='config',
        palette='viridis',
        capsize=0.1
    )
    plt.title('Average Detection Count by Sensor & Configuration')
    plt.ylabel('Detections (mean)')
    plt.xlabel('Sensor')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detections_barplot.png", dpi=150)
    plt.close()
    print(f"   ✓ Saved: detections_barplot.png")


def plot_points_in_comparison(df, output_dir="plots"):
    """Visualize input point cloud sizes."""
    Path(output_dir).mkdir(exist_ok=True)
    
    points_df = df[df['metric_name'] == 'points_in'].copy()
    
    if points_df.empty:
        print("⚠ No points_in data found")
        return
    
    valid_df = points_df.dropna(subset=['sensor', 'config'])
    if not valid_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=valid_df,
            x='config',
            y='value',
            hue='sensor',
            palette='coolwarm'
        )
        plt.title('Input Point Cloud Size Distribution')
        plt.ylabel('Points Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/points_in_boxplot.png", dpi=150)
        plt.close()
        print(f"   ✓ Saved: points_in_boxplot.png")


def generate_summary_report(df, output_file="metrics_summary.txt"):
    """Generate a text summary of key statistics."""
    
    with open(output_file, 'w') as f:
        f.write("📊 LiDAR Metrics Summary Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Configurations: {df['config'].dropna().unique().tolist()}\n")
        f.write(f"Sensors: {df['sensor'].dropna().unique().tolist()}\n")
        f.write(f"Metrics: {df['metric_name'].dropna().unique().tolist()}\n\n")
        
        for config in df['config'].dropna().unique():
            f.write(f"\n🔧 Configuration: {config}\n")
            f.write("-"*40 + "\n")
            
            config_df = df[df['config'] == config]
            
            for metric in ['processing_ms', 'detections_count', 'points_in']:
                metric_df = config_df[config_df['metric_name'] == metric]
                if not metric_df.empty:
                    f.write(f"\n📈 {metric}:\n")
                    for sensor in metric_df['sensor'].dropna().unique():
                        if pd.isna(sensor):
                            continue
                        sensor_data = metric_df[metric_df['sensor'] == sensor]['value']
                        if len(sensor_data) > 0:
                            f.write(f"   • {str(sensor):8s}: mean={sensor_data.mean():8.2f}, "
                                  f"std={sensor_data.std():8.2f}, "
                                  f"n={len(sensor_data)}\n")
    
    print(f"   ✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize LiDAR metrics from ROS CSV exports')
    parser.add_argument('--data-dir', default='.', help='Directory containing CSV files')
    parser.add_argument('--pattern', default='metrics_*.csv', help='File pattern to match')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 LiDAR Metrics Visualizer")
    print("="*60)
    print(f"📁 Data directory: {os.path.abspath(args.data_dir)}")
    print(f"📄 File pattern: {args.pattern}")
    print(f"📊 Output directory: {args.output_dir}")
    print("="*60)
    
    print("\n🔍 Loading LiDAR metrics files...")
    df = load_all_files(args.data_dir, args.pattern)
    
    if df.empty:
        print("\n❌ No data loaded!")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check files exist: ls -lh " + os.path.join(args.data_dir, args.pattern))
        print("   2. Check file content: head -5 " + os.path.join(args.data_dir, "metrics_*.csv"))
        return
    
    print(f"\n{'='*60}")
    print(f"📊 Total records loaded: {len(df):,}")
    print(f"   Configurations: {df['config'].dropna().unique().tolist()}")
    print(f"   Sensors: {df['sensor'].dropna().unique().tolist()}")
    print(f"   Metrics: {df['metric_name'].dropna().unique().tolist()}")
    
    print("\n🎨 Generating visualizations...")
    plot_processing_time_comparison(df, args.output_dir)
    plot_detections_comparison(df, args.output_dir)
    plot_points_in_comparison(df, args.output_dir)
    generate_summary_report(df, os.path.join(args.output_dir, "metrics_summary.txt"))
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Check the '{args.output_dir}' folder for plots.")
    print("="*60)


if __name__ == "__main__":
    main()
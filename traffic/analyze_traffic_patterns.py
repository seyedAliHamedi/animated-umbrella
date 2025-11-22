#!/usr/bin/env python3
"""
Analyze traffic patterns in MAWI CSV by day of week and hour of day.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def analyze_traffic_patterns(csv_path: str = "./timestamps/TL_MAWI-WIDE_2023-2025.csv"):
    """Analyze Total/T traffic distribution by days of week and hours."""
    
    # Load data
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert date and time to datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['weekday'] = df['datetime'].dt.weekday  # 0=Monday, 6=Sunday
    
    # Remove rows with 0 or NaN Total/T
    df_clean = df[df['Total/T'].notna() & (df['Total/T'] > 0)]
    
    print(f"Total rows: {len(df)}")
    print(f"Valid traffic rows: {len(df_clean)}")
    print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print()
    
    # Traffic distribution by day of week
    print("=== TRAFFIC BY DAY OF WEEK ===")
    daily_stats = df_clean.groupby('day_of_week')['Total/T'].agg(['mean', 'median', 'std', 'count'])
    
    # Reorder by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats = daily_stats.reindex(day_order)
    
    print(daily_stats)
    print()
    
    # Rank days by average traffic
    ranked_days = daily_stats.sort_values('mean', ascending=False)
    print("Days ranked by average traffic (highest to lowest):")
    for i, (day, stats) in enumerate(ranked_days.iterrows(), 1):
        print(f"{i}. {day}: {stats['mean']:.1f} avg, {stats['median']:.1f} median")
    print()
    
    # Traffic distribution by hour of day
    print("=== TRAFFIC BY HOUR OF DAY ===")
    hourly_stats = df_clean.groupby('hour')['Total/T'].agg(['mean', 'median', 'std', 'count'])
    print(hourly_stats)
    print()
    
    # Categorize hours by traffic level
    hourly_means = hourly_stats['mean']
    
    # Define thresholds (using quartiles)
    q1 = hourly_means.quantile(0.25)
    q3 = hourly_means.quantile(0.75)
    
    low_traffic = hourly_means[hourly_means <= q1]
    mild_traffic = hourly_means[(hourly_means > q1) & (hourly_means < q3)]
    heavy_traffic = hourly_means[hourly_means >= q3]
    
    print("TRAFFIC CATEGORIZATION BY HOUR:")
    print(f"Low traffic hours (≤ {q1:.1f}): {list(low_traffic.index)}")
    print(f"Mild traffic hours ({q1:.1f} - {q3:.1f}): {list(mild_traffic.index)}")
    print(f"Heavy traffic hours (≥ {q3:.1f}): {list(heavy_traffic.index)}")
    print()
    
    # Detailed hour ranking
    ranked_hours = hourly_stats.sort_values('mean', ascending=False)
    print("Hours ranked by average traffic (highest to lowest):")
    for i, (hour, stats) in enumerate(ranked_hours.iterrows(), 1):
        category = "HEAVY" if hour in heavy_traffic.index else "MILD" if hour in mild_traffic.index else "LOW"
        print(f"{i:2d}. {hour:02d}:00 - {stats['mean']:6.1f} avg ({category})")
    print()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Day of week plot
    daily_stats.reindex(day_order)['mean'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Average Traffic by Day of Week')
    axes[0,0].set_ylabel('Average Total/T')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Hour of day plot
    hourly_stats['mean'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Average Traffic by Hour of Day')
    axes[0,1].set_ylabel('Average Total/T')
    axes[0,1].set_xlabel('Hour')
    
    # Heatmap: Day vs Hour
    pivot_table = df_clean.pivot_table(values='Total/T', index='day_of_week', columns='hour', aggfunc='mean')
    pivot_table = pivot_table.reindex(day_order)
    
    sns.heatmap(pivot_table, ax=axes[1,0], cmap='YlOrRd', cbar_kws={'label': 'Average Total/T'})
    axes[1,0].set_title('Traffic Heatmap: Day vs Hour')
    axes[1,0].set_ylabel('Day of Week')
    axes[1,0].set_xlabel('Hour of Day')
    
    # Box plot by traffic level
    df_clean['traffic_category'] = df_clean['traffic_level'].map({
        1: 'Level 1', 2: 'Level 2', 3: 'Level 3', 4: 'Level 4'
    })
    
    df_clean.boxplot(column='Total/T', by='traffic_category', ax=axes[1,1])
    axes[1,1].set_title('Traffic Distribution by Traffic Level')
    axes[1,1].set_ylabel('Total/T')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'traffic_analysis.png'")
    
    return daily_stats, hourly_stats

if __name__ == "__main__":
    daily_stats, hourly_stats = analyze_traffic_patterns()
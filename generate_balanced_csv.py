#!/usr/bin/env python3
"""
Script to generate a balanced CSV from MAWI dataset with configurable traffic level distribution.
Each 100-row block will have the exact specified distribution of traffic levels.
"""

import pandas as pd
from typing import Dict

def generate_balanced_csv(
    input_csv_path: str = "./timestamps/TL_MAWI-WIDE_2023-2025.csv",
    output_csv_path: str = "./timestamps/TL_MAWI_balanced.csv",
    total_epochs: int = 10000,
    block_size: int = 100,
    traffic_distribution: Dict[int, float] = {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.1}
):
    """
    Generate a balanced CSV with specified traffic level distribution per block.
    
    Args:
        input_csv_path: Path to original MAWI CSV
        output_csv_path: Path for output balanced CSV
        total_epochs: Total number of epochs (rows) to generate
        block_size: Size of each distribution block (default 100)
        traffic_distribution: Dict mapping traffic_level to proportion (must sum to 1.0)
    """
    
    # Validate distribution
    if not abs(sum(traffic_distribution.values()) - 1.0) < 1e-10:
        raise ValueError(f"Traffic distribution must sum to 1.0, got {sum(traffic_distribution.values())}")
    
    # Load original data
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Group by traffic level
    traffic_groups = {}
    for level in traffic_distribution.keys():
        traffic_groups[level] = df[df['traffic_level'] == level].copy()
        print(f"Traffic level {level}: {len(traffic_groups[level])} rows available")
    
    # Calculate rows per traffic level per block
    rows_per_level = {}
    for level, proportion in traffic_distribution.items():
        rows_per_level[level] = int(block_size * proportion)
    
    # Adjust for rounding errors - add remaining rows to the largest group
    total_assigned = sum(rows_per_level.values())
    if total_assigned < block_size:
        max_level = max(traffic_distribution.items(), key=lambda x: x[1])[0]
        rows_per_level[max_level] += block_size - total_assigned
    
    print(f"\nRows per {block_size}-epoch block:")
    for level, count in rows_per_level.items():
        print(f"  Traffic level {level}: {count} rows ({count/block_size*100:.1f}%)")
    
    # Generate balanced dataset
    num_blocks = total_epochs // block_size
    print(f"\nGenerating {num_blocks} blocks of {block_size} epochs each...")
    
    balanced_rows = []
    
    for block_idx in range(num_blocks):
        print(f"Generating block {block_idx + 1}/{num_blocks}...")
        block_rows = []
        
        # Sample required rows for each traffic level
        for level, count in rows_per_level.items():
            available_rows = traffic_groups[level]
            if len(available_rows) < count:
                print(f"Warning: Only {len(available_rows)} rows available for traffic level {level}, need {count}")
                sampled = available_rows.sample(n=len(available_rows), replace=False)
                # Fill remaining with replacement
                remaining = count - len(available_rows)
                if remaining > 0:
                    sampled_extra = available_rows.sample(n=remaining, replace=True)
                    sampled = pd.concat([sampled, sampled_extra], ignore_index=True)
            else:
                sampled = available_rows.sample(n=count, replace=False)
            
            block_rows.append(sampled)
        
        # Combine and shuffle the block
        block_df = pd.concat(block_rows, ignore_index=True)
        block_df = block_df.sample(frac=1.0).reset_index(drop=True)  # Shuffle
        
        balanced_rows.append(block_df)
    
    # Handle remaining epochs (if total_epochs is not divisible by block_size)
    remaining_epochs = total_epochs % block_size
    if remaining_epochs > 0:
        print(f"Generating final {remaining_epochs} rows...")
        final_rows = []
        for level, proportion in traffic_distribution.items():
            count = int(remaining_epochs * proportion)
            if count > 0:
                sampled = traffic_groups[level].sample(n=count, replace=False)
                final_rows.append(sampled)
        
        if final_rows:
            final_df = pd.concat(final_rows, ignore_index=True)
            final_df = final_df.sample(frac=1.0).reset_index(drop=True)
            balanced_rows.append(final_df)
    
    # Combine all blocks
    result_df = pd.concat(balanced_rows, ignore_index=True)
    
    # Save to file
    print(f"\nSaving balanced CSV with {len(result_df)} rows to {output_csv_path}...")
    result_df.to_csv(output_csv_path, index=False)
    
    # Verify the result
    print("\nVerification - Traffic level distribution per block:")
    for i in range(min(3, num_blocks)):  # Show first 3 blocks
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, len(result_df))
        block_data = result_df.iloc[start_idx:end_idx]
        distribution = block_data['traffic_level'].value_counts().sort_index()
        print(f"  Block {i+1} ({start_idx}-{end_idx-1}): {distribution.to_dict()}")
    
    print(f"\nOverall distribution:")
    overall_dist = result_df['traffic_level'].value_counts().sort_index()
    for level, count in overall_dist.items():
        print(f"  Traffic level {level}: {count} rows ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nBalanced CSV generated successfully: {output_csv_path}")
    return result_df


if __name__ == "__main__":
    # Configuration
    TOTAL_EPOCHS = 10000
    BLOCK_SIZE = 100
    TRAFFIC_DISTRIBUTION = {
        1: 0.6,  
        2: 0.20,  
        3: 0.1, 
        4: 0.1   
    }
    
    # Generate balanced CSV
    generate_balanced_csv(
        total_epochs=TOTAL_EPOCHS,
        block_size=BLOCK_SIZE,
        traffic_distribution=TRAFFIC_DISTRIBUTION
    )
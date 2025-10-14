import arcpy
import pandas as pd
from typing import List, Dict, Tuple
import os


def create_density_table(out_tb: str) -> None:
    """Create a density table with required fields"""
    tb_path = os.path.dirname(out_tb)
    tb_name = os.path.basename(out_tb)

    # Create table
    arcpy.management.CreateTable(tb_path, tb_name)

    # Add fields
    field_definitions = [
        ("group_id", "LONG"),
        ("dt", "DATE"),
        ("seq", "LONG"),
        ("density_val", "DOUBLE")
    ]
    
    arcpy.AddMessage("Adding fields to the table...")
    for field_name, field_type in field_definitions:
        arcpy.management.AddField(out_tb, field_name, field_type)

    arcpy.AddMessage(f"Table created successfully: {out_tb}")


def extract_density_sequences(ts: List[float], min_val: float, max_val: float) -> List[Dict[int, float]]:
    """
    Extract sequences that meet density criteria and return element indices
    
    Args:
        ts: List of time series values
        min_val: Minimum threshold value
        max_val: Maximum threshold value
        
    Returns:
        List of dictionaries containing indices and values
    """
    results = []
    current = []
    
    for idx, x in enumerate(ts):
        if min_val <= x < max_val:
            current.append((x, idx+1))
        else:
            if current:
                # Convert current to {idx: x, ...}
                results.append({idx: x for x, idx in current})
                current = []
                
    if current:
        results.append({idx: x for x, idx in current})
        
    return results


def get_data_from_table(input_table: str, fields: List[str]) -> pd.DataFrame:
    """
    Read data from ArcGIS table and return as DataFrame
    
    Args:
        input_table: Input table path
        fields: List of field names to read
        
    Returns:
        DataFrame containing the table data
    """
    arcpy.AddMessage("Reading data from input table...")
    
    # Use SearchCursor to read data
    data = [row for row in arcpy.da.SearchCursor(input_table, fields)]
    
    arcpy.AddMessage(f"Successfully read {len(data)} records from the input table")
    return pd.DataFrame(data, columns=fields)


def calculate_density_values(df: pd.DataFrame, group_field: str, 
                            sequence_field: str, value_field: str, 
                            time_field: str,
                            min_val: float, max_val: float) -> Tuple[Dict, Dict]:
    """
    Calculate density values and return density dictionary and sequence time dictionary
    
    Args:
        df: DataFrame with input data
        group_field: Field name for grouping
        sequence_field: Field name for sequence
        value_field: Field name for values
        time_field: Field name for time
        min_val: Minimum threshold value
        max_val: Maximum threshold value
        
    Returns:
        Tuple containing density dictionary and sequence-time mapping dictionary
    """
    arcpy.AddMessage("Calculating density values...")
    
    # Get sequences from first group and sort
    first_group = df[group_field].iloc[0]
    seq_list = df.loc[df[group_field] == first_group, sequence_field].sort_values().tolist()
    
    # Create sequence-time mapping dictionary
    subset = df[df[group_field] == first_group]
    subset = subset[[sequence_field, time_field]].sort_values(by=sequence_field)
    seq_time_dict = dict(zip(subset[sequence_field], subset[time_field]))
    
    # Initialize density dictionary for each group
    group_values = df[group_field].unique()
    density_dic = {g: {seq: 0 for seq in seq_list} for g in group_values}
    
    # Group data and calculate densities
    grouped_data = (
        df.groupby(group_field)
        .apply(lambda g: g.sort_values(sequence_field)[value_field].tolist())
        .to_dict()
    )
    
    # Set up progress reporting
    total_groups = len(grouped_data)
    arcpy.SetProgressor("step", "Processing groups...", 0, total_groups, 1)
    
    for i, (group_id, values) in enumerate(grouped_data.items()):
        sequences = extract_density_sequences(values, min_val, max_val)
        
        for sequence in sequences:
            for seq_id, _ in sequence.items():
                density_dic[group_id][seq_id] += 1
        
        # Update progress
        arcpy.SetProgressorPosition(i + 1)
        if (i + 1) % max(1, total_groups // 10) == 0:  # Report every 10% of progress
            arcpy.AddMessage(f"Processed {i + 1} of {total_groups} groups ({((i + 1) / total_groups) * 100:.1f}%)")
    
    arcpy.ResetProgressor()
    arcpy.AddMessage("Density calculations completed")
    return density_dic, seq_time_dict


def insert_data_to_table(out_tb: str, density_dic: Dict, seq_time_dict: Dict) -> None:
    """
    Insert calculated data into the output table
    
    Args:
        out_tb: Output table path
        density_dic: Dictionary with density values
        seq_time_dict: Dictionary mapping sequence IDs to times
    """
    arcpy.AddMessage("Inserting data into output table...")
    
    # Count total rows to insert
    total_rows = sum(len(values) for values in density_dic.values())
    arcpy.SetProgressor("step", "Inserting rows...", 0, total_rows, 1)
    
    row_count = 0
    with arcpy.da.InsertCursor(out_tb, ["group_id", "dt", "seq", "density_val"]) as cursor:
        for group_id, density_values in density_dic.items():
            for seq_id, density_value in density_values.items():
                cursor.insertRow([group_id, seq_time_dict[seq_id], seq_id, density_value])
                row_count += 1
                
                # Update progress
                arcpy.SetProgressorPosition(row_count)
                if row_count % max(1, total_rows // 20) == 0:  # Report every 5% of progress
                    arcpy.AddMessage(f"Inserted {row_count} of {total_rows} rows ({(row_count / total_rows) * 100:.1f}%)")
    
    arcpy.ResetProgressor()
    arcpy.AddMessage(f"All data inserted successfully: {row_count} rows")


def script_tool(input_sts_table: str, time_field: str, value_field: str, 
               sequence_field: str, group_field: str, out_tb: str, 
               min_density: float = 70.0, max_density: float = 120.0) -> None:
    """
    Main script tool function
    
    Args:
        input_sts_table: Input table path
        time_field: Field name for time values
        value_field: Field name for density values
        sequence_field: Field name for sequence identifiers
        group_field: Field name for group identifiers
        out_tb: Output table path
        min_density: Minimum density threshold (default: 70.0)
        max_density: Maximum density threshold (default: 120.0)
    """
    arcpy.AddMessage("Starting density analysis tool...")
    
    # Read data
    fields = [time_field, value_field, sequence_field, group_field]
    df = get_data_from_table(input_sts_table, fields)
    
    # Calculate density values
    density_dic, seq_time_dict = calculate_density_values(
        df, group_field, sequence_field, value_field, time_field,
        min_density, max_density
    )
    
    # Create output table
    create_density_table(out_tb)
    
    # Insert data
    insert_data_to_table(out_tb, density_dic, seq_time_dict)
    
    arcpy.AddMessage("Density analysis completed successfully")


if __name__ == "__main__":
    # Get tool parameters
    input_sts_table = arcpy.GetParameterAsText(0)
    time_field = arcpy.GetParameterAsText(1)
    value_field = arcpy.GetParameterAsText(2)
    sequence_field = arcpy.GetParameterAsText(3)
    group_field = arcpy.GetParameterAsText(4)
    out_tb = arcpy.GetParameterAsText(5)
    
    # Run tool
    arcpy.AddMessage("Initializing density analysis tool...")
    script_tool(input_sts_table, time_field, value_field, sequence_field, group_field, out_tb)
    arcpy.AddMessage("Tool execution completed")
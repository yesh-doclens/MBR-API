import pandas as pd
import re
from pathlib import Path
from typing import Tuple, Dict, Any

def load_medicare_procedure_prices() -> pd.DataFrame:
    """Load the CPT-Procedure-Prices-Medicare CSV file."""
    csv_path = Path(__file__).parent.parent.parent / "CPT-Procedure-Prices-Medicare.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()

def load_medicare_drug_prices() -> pd.DataFrame:
    """Load the Drug-Prices-Medicare CSV file."""
    csv_path = Path(__file__).parent.parent.parent / "Drug-Prices-Medicare.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()

def parse_dosage_units(dosage_str: str) -> Tuple[float, str]:
    """Parse the HCPCS Code Dosage string to extract the numeric value and unit."""
    if not dosage_str or pd.isna(dosage_str):
        return 1.0, ""
    
    dosage_str = str(dosage_str).strip()
    match = re.match(r'^([\d.]+)\s*(.*)$', dosage_str)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2).strip().upper()
            return value, unit
        except ValueError:
            return 1.0, dosage_str
    return 1.0, dosage_str

def enrich_with_medicare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add Medicare threshold columns to the DataFrame."""
    procedure_prices_df = load_medicare_procedure_prices()
    drug_prices_df = load_medicare_drug_prices()
    
    procedure_lookup = {}
    if not procedure_prices_df.empty and 'HCPCS' in procedure_prices_df.columns:
        for _, row in procedure_prices_df.iterrows():
            hcpcs = str(row.get('HCPCS', '')).strip().upper()
            non_fac_total = row.get('NON-FACILITY TOTAL PRICE', '')
            if hcpcs and non_fac_total:
                try:
                    procedure_lookup[hcpcs] = float(str(non_fac_total).replace(',', ''))
                except ValueError:
                    pass
    
    drug_lookup = {}
    if not drug_prices_df.empty and 'HCPCS Code' in drug_prices_df.columns:
        for _, row in drug_prices_df.iterrows():
            hcpcs = str(row.get('HCPCS Code', '')).strip().upper()
            payment_limit = row.get('Payment Limit', '')
            dosage = row.get('HCPCS Code Dosage', '1')
            if hcpcs and payment_limit:
                try:
                    drug_lookup[hcpcs] = {
                        'payment_limit': float(str(payment_limit).replace(',', '')),
                        'dosage': dosage
                    }
                except ValueError:
                    pass
    
    unit_threshold_prices = []
    medicare_units_list = []
    total_threshold_prices = []
    
    for _, row in df.iterrows():
        cpt_code = str(row.get('CPT Code', '')).strip().upper()
        entity_type = str(row.get('Entity Type', '')).lower()
        quantity = int(row.get('Quantity', 1))
        dosage_value = row.get('Dosage Value')
        dosage_unit = str(row.get('Dosage Unit', '')).strip().upper()
        num_units_from_name = row.get('Units from Name')
        unit_threshold = 0.0
        medicare_units = 0.0
        
        if entity_type == 'procedure':
            if cpt_code in procedure_lookup:
                unit_threshold = procedure_lookup[cpt_code]
                medicare_units = float(quantity)
        
        elif entity_type == 'drug':
            if cpt_code in drug_lookup:
                drug_info = drug_lookup[cpt_code]
                unit_threshold = drug_info['payment_limit']
                med_dosage_val, med_dosage_unit = parse_dosage_units(drug_info['dosage'])
                
                if dosage_value and dosage_unit == med_dosage_unit:
                    medicare_units = ((quantity * dosage_value * (num_units_from_name if num_units_from_name else 1.0)) / med_dosage_val) if med_dosage_val != 0 else (quantity * (num_units_from_name if num_units_from_name else 1.0))
                else:
                    medicare_units = 1.0 * (num_units_from_name if num_units_from_name else 1.0) * quantity
        
        total_threshold = unit_threshold * medicare_units
        unit_threshold_prices.append(round(unit_threshold, 2))
        medicare_units_list.append(round(medicare_units, 2))
        total_threshold_prices.append(round(total_threshold, 2))
    
    df['Unit Price Threshold'] = unit_threshold_prices
    df['No. of Medicare Units'] = medicare_units_list
    df['Total Threshold Price'] = total_threshold_prices
    
    df['Threshold Multiplier'] = df.apply(
        lambda r: r['Total Price'] / r['Total Threshold Price'] if r['Total Threshold Price'] > 0 else None, 
        axis=1
    )
    df['Price Difference'] = df['Total Price'] - df['Total Threshold Price']
    
    return df

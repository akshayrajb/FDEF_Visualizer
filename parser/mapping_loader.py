"""
Mapping Sheet Loader for FDEF Dependency Visualizer

Reads Excel mapping sheets with columns:
- Column C: Network Line
- Column D: Network Name  
- Column F: Internal Name

Provides bidirectional mapping between network and internal signal names.
"""

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MappingLoader:
    """
    Loads and processes Excel mapping sheets for signal name translation.
    
    Expected Excel format:
    - Column C: Network Line (identifier)
    - Column D: Network Name (external signal name)
    - Column F: Internal Name (internal signal name)
    """
    
    def __init__(self, excel_path: Path):
        """
        Initialize the mapping loader.
        
        Args:
            excel_path: Path to the Excel mapping file
        """
        self.excel_path = Path(excel_path)
        self.by_network = {}  # Network name -> Internal name
        self.by_internal = {}  # Internal name -> Network name
        self.mapping_df = None
        
    def load(self) -> pd.DataFrame:
        """
        Load the mapping sheet and create lookup dictionaries.
        
        Returns:
            DataFrame with mapping data
            
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.excel_path}")
        
        try:
            # Read Excel file with specific columns
            df = pd.read_excel(
                self.excel_path,
                usecols=["C", "D", "F"],  # Network Line | Network Name | Internal Name
                header=None,
                skiprows=1,  # Skip header row
                names=["network_line", "network_name", "internal_name"]
            )
            
            # Remove completely empty rows
            df = df.dropna(how="all")
            
            # Remove rows where both network_name and internal_name are empty
            df = df.dropna(subset=["network_name", "internal_name"], how="all")
            
            # Fill NaN values with empty strings for consistency
            df = df.fillna("")
            
            # Create bidirectional lookup dictionaries
            self._create_lookup_dicts(df)
            
            self.mapping_df = df
            logger.info(f"Loaded {len(df)} mapping entries from {self.excel_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading mapping file {self.excel_path}: {e}")
            raise ValueError(f"Failed to load mapping file: {e}")
    
    def _create_lookup_dicts(self, df: pd.DataFrame):
        """
        Create bidirectional lookup dictionaries from the mapping DataFrame.
        
        Args:
            df: DataFrame with mapping data
        """
        # Clear existing mappings
        self.by_network.clear()
        self.by_internal.clear()
        
        for _, row in df.iterrows():
            network_name = str(row["network_name"]).strip()
            internal_name = str(row["internal_name"]).strip()
            
            # Only create mappings for non-empty values
            if network_name and internal_name:
                self.by_network[network_name] = internal_name
                self.by_internal[internal_name] = network_name
    
    def get_internal_name(self, network_name: str) -> str:
        """
        Get internal signal name from network name.
        
        Args:
            network_name: Network signal name
            
        Returns:
            Internal signal name, or original name if not found
        """
        return self.by_network.get(network_name, network_name)
    
    def get_network_name(self, internal_name: str) -> str:
        """
        Get network signal name from internal name.
        
        Args:
            internal_name: Internal signal name
            
        Returns:
            Network signal name, or original name if not found
        """
        return self.by_internal.get(internal_name, internal_name)
    
    def resolve_signal_name(self, signal_name: str) -> tuple[str, str]:
        """
        Resolve signal name to both network and internal names.
        
        Args:
            signal_name: Signal name (could be network or internal)
            
        Returns:
            Tuple of (network_name, internal_name)
        """
        # Check if it's a network name
        if signal_name in self.by_network:
            return signal_name, self.by_network[signal_name]
        
        # Check if it's an internal name
        if signal_name in self.by_internal:
            return self.by_internal[signal_name], signal_name
        
        # Not found in mapping, return as-is
        return signal_name, signal_name
    
    def search_signals(self, pattern: str, case_sensitive: bool = False) -> list[dict]:
        """
        Search for signals matching a pattern.
        
        Args:
            pattern: Search pattern (can include wildcards)
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching signal dictionaries
        """
        if not self.mapping_df is not None:
            return []
        
        results = []
        search_pattern = pattern if case_sensitive else pattern.lower()
        
        for _, row in self.mapping_df.iterrows():
            network_name = str(row["network_name"]).strip()
            internal_name = str(row["internal_name"]).strip()
            
            # Prepare search strings
            search_network = network_name if case_sensitive else network_name.lower()
            search_internal = internal_name if case_sensitive else internal_name.lower()
            
            # Check if pattern matches
            if (search_pattern in search_network or 
                search_pattern in search_internal):
                results.append({
                    "network_line": row["network_line"],
                    "network_name": network_name,
                    "internal_name": internal_name
                })
        
        return results
    
    def get_all_signals(self) -> list[str]:
        """
        Get all unique signal names (both network and internal).
        
        Returns:
            List of all signal names
        """
        signals = set()
        signals.update(self.by_network.keys())
        signals.update(self.by_internal.keys())
        return sorted(list(signals))
    
    def get_mapping_stats(self) -> dict:
        """
        Get statistics about the loaded mapping.
        
        Returns:
            Dictionary with mapping statistics
        """
        if self.mapping_df is None:
            return {"total_entries": 0, "network_signals": 0, "internal_signals": 0}
        
        return {
            "total_entries": len(self.mapping_df),
            "network_signals": len(self.by_network),
            "internal_signals": len(self.by_internal),
            "file_path": str(self.excel_path)
        }
    
    def validate_mapping(self) -> list[str]:
        """
        Validate the loaded mapping for common issues.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if self.mapping_df is None:
            warnings.append("No mapping data loaded")
            return warnings
        
        # Check for duplicate network names
        network_names = [name for name in self.by_network.keys() if name]
        if len(network_names) != len(set(network_names)):
            warnings.append("Duplicate network names found")
        
        # Check for duplicate internal names
        internal_names = [name for name in self.by_internal.keys() if name]
        if len(internal_names) != len(set(internal_names)):
            warnings.append("Duplicate internal names found")
        
        # Check for empty mappings
        empty_network = sum(1 for _, row in self.mapping_df.iterrows() 
                          if not str(row["network_name"]).strip())
        empty_internal = sum(1 for _, row in self.mapping_df.iterrows() 
                           if not str(row["internal_name"]).strip())
        
        if empty_network > 0:
            warnings.append(f"{empty_network} rows with empty network names")
        if empty_internal > 0:
            warnings.append(f"{empty_internal} rows with empty internal names")
        
        return warnings
from dataclasses import dataclass


# Output artifact from the data ingestion process.
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

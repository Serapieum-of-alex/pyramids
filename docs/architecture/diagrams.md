# Architecture Diagrams

Below are core diagrams describing the system at multiple levels.

## C4: System Context

```mermaid
flowchart LR
  user(User) -->|Provides GIS data paths & commands| pyramids{{pyramids package}}
  ext1[(Raster files\nGeoTIFF/ASC/NetCDF)] --> pyramids
  ext2[(Vector files\nShapefile/GeoJSON/GPKG)] --> pyramids
  pyramids --> out1[(Processed rasters\nGeoTIFF/ASC)]
  pyramids --> out2[(Processed vectors\nGeoJSON/GPKG)]
```

## C4: Containers

```mermaid
flowchart TB
  subgraph Runtime Process
    A[Dataset]:::c --> B[_io]
    C[Datacube]:::c --> B
    D[FeatureCollection]:::c --> B
    A --> E[_utils]
    C --> E
    D --> E
  end
  classDef c fill:#eef,stroke:#88f
```

## C4: Components

```mermaid
flowchart LR
  io[_io: read_file, to_ascii, path parsing]
  utils[_utils: geometry/index helpers]
  ds[dataset.Dataset]
  abs[abstract_dataset.AbstractDataset]
  dc[datacube.Datacube]
  fc[featurecollection.FeatureCollection]

  abs --> ds
  ds --> io
  dc --> ds
  fc --> io
  ds --> utils
  fc --> utils
```

## UML Class: Raster Core

```mermaid
classDiagram
  class AbstractDataset {
    <<abstract>>
    +read_file(path, read_only)
    +to_file(path, band)
  }
  class Dataset {
    +read_file(path, read_only, file_i)
    +to_file(path, band, tile_length)
    +read()
  }
  AbstractDataset <|-- Dataset
```

## UML Class: Vector Core

```mermaid
classDiagram
  class FeatureCollection {
    +read_file(path)
    +to_file(path, driver)
  }
```

## Sequence: Read Raster from Zip

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  participant IO as _io
  U->>DS: Dataset.read_file("dem.zip!dem.tif")
  DS->>IO: _parse_path(path)
  IO-->>DS: zip path + inner file
  DS->>IO: read_file(...)
  IO-->>DS: array + meta
  DS-->>U: Dataset instance
```

## Sequence: Save Raster to GeoTIFF

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  U->>DS: to_file("out.tif")
  DS-->>U: writes file
```

## Sequence: Build Datacube from Folder

```mermaid
sequenceDiagram
  participant U as User
  participant DC as Datacube
  participant DS as Dataset
  U->>DC: from_folder("./rasters/*.tif")
  DC->>DS: open each file
  DS-->>DC: Dataset objects
  DC-->>U: Datacube
```

## Sequence: Zonal Statistics

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  participant FC as FeatureCollection
  U->>FC: read_file(polygons.gpkg)
  U->>DS: read_file(raster.tif)
  FC->>DS: zonal_stats(raster)
  DS-->>U: table
```

## Sequence: Align and Resample

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  U->>DS: align_to(reference)
  DS-->>U: aligned dataset
```

## Dependency Graph (Modules)

```mermaid
flowchart LR
  abstract_dataset --> dataset
  _io --> dataset
  _utils --> dataset
  dataset --> datacube
  _io --> featurecollection
  _utils --> featurecollection
```

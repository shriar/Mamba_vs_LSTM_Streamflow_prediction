"""CAMELS data download, scaling, and loading utilities."""

import os
import pickle
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GDRIVE_FILE_IDS = {
    "training_file": "1z9UBkn86dBIKo1wu1v7bYSTHG87RzGWX",
    "validation_file": "12f-ZrWLxtWRwzacmxBZ23-76_buGhlNa",
}


def download_file(url, destination):
    """Stream a URL to a local file."""
    import requests

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


def download_google_drive_file(file_id, destination, chunk_size=1024 * 1024):
    """Download a file from Google Drive, handling large file virus-scan prompts.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    destination : str or Path
        Target local file path.
    chunk_size : int, optional
        Streaming chunk size in bytes (default 1MB).
    """
    import re
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    session = requests.Session()
    url = f"https://drive.google.com/uc?id={file_id}"
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()

    # Google Drive redirects large files (>100MB) to an HTML virus-scan confirmation form
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        text = response.text
        form_action = re.search(r'action="([^"]+)"', text)
        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', text)
        uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', text)

        if form_action and confirm_match and uuid_match:
            download_url = form_action.group(1)
            params = {
                "id": file_id,
                "confirm": confirm_match.group(1),
                "uuid": uuid_match.group(1),
            }
            response = session.get(download_url, params=params, stream=True, timeout=120)
            response.raise_for_status()
        else:
            # Fallback for alternative warning formats
            token = None
            for key, val in response.cookies.items():
                if key.startswith("download_warning"):
                    token = val
                    break
            if token:
                response = session.get(
                    "https://drive.google.com/uc?export=download",
                    params={"id": file_id, "confirm": token},
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()
            else:
                m_href = re.search(r'href="(/uc\?[^"]*confirm=[^"]*)"', text)
                if m_href:
                    confirm_url = "https://drive.google.com" + m_href.group(1).replace("&amp;", "&")
                    response = session.get(confirm_url, stream=True, timeout=120)
                    response.raise_for_status()
                else:
                    raise RuntimeError(
                        f"Could not parse Google Drive download confirmation page for file ID '{file_id}'."
                    )

    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    size_mb = f"{total_size / (1024 * 1024):.1f} MB" if total_size > 0 else "unknown size"
    print(f"Downloading '{destination.name}' from Google Drive ({size_mb})...")

    try:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        print(f"\r  [{pct:5.1f}%] {mb:.1f} / {tot_mb:.1f} MB", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded {mb:.1f} MB", end="", flush=True)
        if total_size > 0:
            print()
        if temp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file '{destination.name}' is empty.")
        if destination.exists():
            destination.unlink()
        temp_path.rename(destination)
        print(f"Saved: {destination}")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_gauge_ids(url, destination, n_basins=None):
    """Fetch the compact official CAMELS gauge list for cached-array runs.

    Also downloads camels_topo.txt (lat/lon) from the same Zenodo record
    so that gauge_ids.csv includes gauge_name, gauge_lat, and gauge_lon.
    """
    import requests

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        content = response.content

    metadata = pd.read_csv(StringIO(content.decode("utf-8")), sep=";", dtype={"gauge_id": str})
    if "gauge_id" not in metadata.columns:
        raise ValueError("CAMELS gauge metadata did not contain a gauge_id column.")
    ids = np.asarray([format_gauge_id(value) for value in metadata["gauge_id"]])
    if n_basins is not None and len(ids) != n_basins:
        raise ValueError(
            f"Downloaded CAMELS metadata has {len(ids)} IDs but the data have {n_basins} basins."
        )

    # Build an enriched DataFrame with gauge_name, lat, lon
    extras = {"gauge_name": metadata.get("gauge_name")}

    # Fetch topo file for lat/lon (same Zenodo base as the name file)
    topo_url = url.replace("camels_name.txt", "camels_topo.txt")
    try:
        with requests.get(topo_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            topo = pd.read_csv(StringIO(resp.content.decode("utf-8")), sep=";",
                               dtype={"gauge_id": str})
        topo["gauge_id"] = topo["gauge_id"].apply(format_gauge_id)
        topo = topo.set_index("gauge_id")
        extras["gauge_lat"] = [topo.loc[gid, "gauge_lat"] if gid in topo.index else np.nan for gid in ids]
        extras["gauge_lon"] = [topo.loc[gid, "gauge_lon"] if gid in topo.index else np.nan for gid in ids]
    except Exception:
        pass  # lat/lon are optional; proceed without them

    return save_gauge_ids(destination, ids, extras=extras)

def download_camels():
    """Download the CAMELS inputs using the layout expected by hydroDL."""
    base_path = Path.cwd()
    archive_base = "https://zenodo.org/records/15529996/files/"
    attributes = base_path / "camels_attributes_v2.0" / "camels_attributes_v2.0"
    attributes.mkdir(parents=True, exist_ok=True)

    dataset_archive = base_path / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    download_file(archive_base + "/basin_timeseries_v1p2_metForcing_obsFlow.zip", dataset_archive)
    with zipfile.ZipFile(dataset_archive) as archive:
        archive.extractall(base_path / "basin_timeseries_v1p2_metForcing_obsFlow")

    for name in (
        "camels_attributes_v2.0.xlsx", "camels_clim.txt", "camels_geol.txt",
        "camels_hydro.txt", "camels_name.txt", "camels_soil.txt",
        "camels_topo.txt", "camels_vege.txt",
    ):
        download_file(archive_base + "/" + name, attributes / name)

    pet_archive = base_path / "pet_harg.zip"
    download_file("https://zenodo.org/record/7943626/files/pet_harg.zip?download=1", pet_archive)
    with zipfile.ZipFile(pet_archive) as archive:
        archive.extractall(base_path)
    print("Download and extraction complete.")


def format_gauge_id(value):
    """Zero-pad USGS IDs so 1013500 becomes 01013500."""
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(8)
    return text


def gauge_ids_from_loader(loader, camels_module=None):
    """Pull basin IDs from a hydroDL DataframeCamels instance."""
    for attr in ("usgsId", "idLst", "gageid", "gageId"):
        ids = getattr(loader, attr, None)
        if ids is not None:
            return np.asarray([format_gauge_id(x) for x in ids])
    if camels_module is not None:
        ids = getattr(camels_module, "gageid", None)
        if ids is not None:
            return np.asarray([format_gauge_id(x) for x in ids])
    raise AttributeError(
        "Could not find gauge IDs on the CAMELS loader "
        "(tried usgsId, idLst, gageid, gageId)."
    )


def save_gauge_ids(path, gauge_ids, extras=None):
    """Write USGS gauge IDs (with optional name/lat/lon) to a CSV keyed by basin row index."""
    path = Path(path)
    ids = [format_gauge_id(x) for x in np.asarray(gauge_ids).reshape(-1)]
    df = pd.DataFrame({"gauge_id": ids})
    if extras:
        for col, values in extras.items():
            if values is not None and len(values) == len(ids):
                df[col] = list(values)
    df.to_csv(path, index_label="basin_index")
    print(f"Saved {len(ids)} gauge IDs to {path}")
    return np.asarray(ids)


def load_gauge_ids(path, n_basins=None):
    """Load USGS gauge IDs previously saved by save_gauge_ids."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gauge ID file not found: {path}. "
            "Re-extract CAMELS (data_opt=1) or create this CSV first."
        )
    df = pd.read_csv(path, dtype={"gauge_id": str})
    if "gauge_id" not in df.columns:
        raise ValueError(f"{path} must contain a gauge_id column.")
    ids = np.asarray([format_gauge_id(x) for x in df["gauge_id"]])
    if n_basins is not None and len(ids) != n_basins:
        raise ValueError(
            f"{path} has {len(ids)} IDs but the data have {n_basins} basins."
        )
    return ids


def extract_camels(t_range, attributes, forcing_variables, camels, *, for_type="daymet",
                   flow_regime=1, subset="All", file_path=None, gauge_id_path=None):
    """Load CAMELS arrays and optionally cache them in a pickle file."""
    loader = camels.DataframeCamels(subset=subset, tRange=t_range, forType=for_type)
    data = (
        loader.getDataTs(varLst=forcing_variables, doNorm=False, rmNan=False, flow_regime=flow_regime),
        loader.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime),
        loader.getDataConst(varLst=attributes, doNorm=False, rmNan=False, flow_regime=flow_regime),
    )
    if file_path is not None:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    gauge_ids = gauge_ids_from_loader(loader, camels)
    if gauge_id_path is not None:
        save_gauge_ids(gauge_id_path, gauge_ids)
    return data, gauge_ids


def _statistics(values, log_transform=False):
    values = values.flatten()
    values = values[~np.isnan(values)]
    if log_transform:
        values = np.log10(np.sqrt(values) + 0.1)
    std = np.std(values).astype(float)
    return [
        np.percentile(values, 10).astype(float),
        np.percentile(values, 90).astype(float),
        np.mean(values).astype(float),
        std if std >= 0.001 else 1,
    ]


def get_statistics(log_norm_columns, *, attribute_names=None, attribute_data=None,
                   series_names=None, series_data=None):
    """Calculate normalization statistics for time series and static attributes."""
    statistics = {}
    if series_names is not None:
        for index, name in enumerate(series_names):
            statistics[name] = _statistics(series_data[:, :, index], name in log_norm_columns)
    if attribute_names is not None:
        for index, name in enumerate(attribute_names):
            statistics[name] = _statistics(attribute_data[:, index])
    return statistics


def transform_by_statistics(data, variables, statistics, log_norm_columns, normalize):
    """Normalize or invert-normalize a 2-D or 3-D array."""
    variables = [variables] if isinstance(variables, str) else variables
    values, output = data.copy(), np.full(data.shape, np.nan)
    is_series = data.ndim == 3
    for index, name in enumerate(variables):
        mean, std = statistics[name][2:]
        target = (slice(None), slice(None), index) if is_series else (slice(None), index)
        if normalize:
            if name in log_norm_columns:
                values[target] = np.log10(np.sqrt(values[target]) + 0.1)
            output[target] = (values[target] - mean) / std
        else:
            output[target] = values[target] * std + mean
            if name in log_norm_columns:
                output[target] = (np.power(10, output[target]) - 0.1) ** 2
    return output


class HydroScaler:
    """CAMELS normalization wrapper with a scikit-learn-like interface."""

    def __init__(self, attrLst, seriesLst, xNanFill, log_norm_cols):
        self.attrLst = attrLst
        self.seriesLst = seriesLst
        self.xNanFill = xNanFill
        self.log_norm_cols = log_norm_cols
        self.stat_dict = None

    def fit(self, attribute_data, series_data):
        self.stat_dict = get_statistics(
            self.log_norm_cols, attribute_names=self.attrLst, attribute_data=attribute_data,
            series_names=self.seriesLst, series_data=series_data,
        )

    def transform(self, data, variables):
        return transform_by_statistics(data, variables, self.stat_dict, self.log_norm_cols, True)

    def fit_transform(self, attribute_data, series_data):
        self.fit(attribute_data, series_data)
        return self.transform(attribute_data, self.attrLst), self.transform(series_data, self.seriesLst)


def basin_norm(values, basin_area, mean_precipitation, normalize):
    """Convert streamflow between ft³/s and dimensionless basin-normalized flow."""
    dimensions = values.ndim
    if dimensions == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    area = np.tile(basin_area, (1, values.shape[1]))
    precipitation = np.tile(mean_precipitation, (1, values.shape[1]))
    conversion = (area * 10**6) * (precipitation * 10**-3)
    flow = values * 0.0283168 * 3600 * 24 / conversion if normalize else values * conversion / (0.0283168 * 3600 * 24)
    return np.expand_dims(flow, axis=2) if dimensions == 3 else flow


# Keep the tutorial's original public names while centralizing their implementation.
getStatDic = get_statistics
calcStat = _statistics


def calcStatgamma(values):
    return _statistics(values, log_transform=True)


def transNormbyDic(x_in, var_lst, stat_dict, log_norm_cols, to_norm):
    return transform_by_statistics(x_in, var_lst, stat_dict, log_norm_cols, to_norm)


def basinNorm(x, basinarea, meanprep, toNorm):
    return basin_norm(x, basinarea, meanprep, toNorm)


def ensure_gauge_ids(cfg, n_basins=None, loader=None, camels_module=None):
    """Load gauge IDs from CSV, or create the CSV from hydroDL on first use."""
    path = cfg.gauge_id_path()
    if path.exists():
        return load_gauge_ids(path, n_basins=n_basins)
    if loader is not None:
        ids = gauge_ids_from_loader(loader, camels_module)
        return save_gauge_ids(path, ids)

    # Preprocessed arrays do not contain their basin IDs.  Fetch the small
    # official name table instead of requiring the full CAMELS time-series
    # archive just to recover this mapping.
    try:
        return download_gauge_ids(cfg.gauge_metadata_url, path, n_basins=n_basins)
    except Exception as metadata_error:
        raise FileNotFoundError(
            f"Gauge ID file not found: {path}. Downloading the official CAMELS "
            f"gauge metadata also failed: {metadata_error}"
        ) from metadata_error



def load_and_scale_data(cfg):
    """Load CAMELS data, fit scaler, and return everything the main scripts need.

    Parameters
    ----------
    cfg : config.ExperimentConfig

    Returns
    -------
    dict with keys:
        train_x, train_y, train_c, val_x, val_y, val_c,
        scaler, basinarea, meanprep, gauge_ids
    """
    from hydroDL.master import default
    from hydroDL.data import camels

    opt_data = default.optDataCamels
    opt_data = default.update(
        opt_data, varT=cfg.varF, varC=cfg.attrLst,
        tRange=cfg.t_train, forType=cfg.for_type, subset=cfg.subset_train,
    )

    # Resolve preprocessed file paths (prefer existing file, otherwise script_dir)
    train_path = getattr(cfg, "train_path", lambda: None)()
    if train_path is None:
        p = Path(cfg.train_file)
        train_path = p if (p.is_absolute() or p.exists()) else cfg.script_dir / cfg.train_file

    val_path = getattr(cfg, "validation_path", lambda: None)()
    if val_path is None:
        p = Path(cfg.validation_file)
        val_path = p if (p.is_absolute() or p.exists()) else cfg.script_dir / cfg.validation_file

    if cfg.data_opt == 1:
        download_camels()
        just_load = False
    elif cfg.data_opt == 2:
        just_load = True
        file_specs = [
            (train_path, getattr(cfg, "train_file_gdrive_id", DEFAULT_GDRIVE_FILE_IDS.get("training_file"))),
            (val_path, getattr(cfg, "validation_file_gdrive_id", DEFAULT_GDRIVE_FILE_IDS.get("validation_file"))),
        ]
        for path, gdrive_id in file_specs:
            if not path.exists():
                if gdrive_id:
                    print(f"Preprocessed CAMELS dataset '{path.name}' not found locally.")
                    print(f"Auto-downloading from Google Drive (ID: {gdrive_id})...")
                    download_google_drive_file(gdrive_id, path)
                else:
                    raise FileNotFoundError(
                        f"Missing preprocessed CAMELS file: {path}. No Google Drive ID specified."
                    )
    else:
        just_load = True

    gauge_id_path = cfg.gauge_id_path()

    if just_load:
        with open(train_path, "rb") as f:
            train_x, train_y, train_c = pickle.load(f)
        with open(val_path, "rb") as f:
            val_x, val_y, val_c = pickle.load(f)
        gauge_ids = ensure_gauge_ids(cfg, n_basins=val_y.shape[0])
    else:
        camels.initcamels(flow_regime=cfg.flow_regime, forType=cfg.for_type,
                          rootDB=str(cfg.script_dir))
        (train_x, train_y, train_c), gauge_ids = extract_camels(
            cfg.t_train, cfg.attrLst, cfg.varF, camels,
            for_type=cfg.for_type, flow_regime=cfg.flow_regime,
            subset=cfg.subset_train, file_path=train_path,
            gauge_id_path=gauge_id_path,
        )
        (val_x, val_y, val_c), _ = extract_camels(
            cfg.t_valid, cfg.attrLst, cfg.varF, camels,
            for_type=cfg.for_type, flow_regime=cfg.flow_regime,
            subset=cfg.subset_train, file_path=val_path,
            gauge_id_path=gauge_id_path,
        )

    # Basin normalization for dimensionless streamflow
    basinarea = val_c[:, np.where(np.array(cfg.attrLst) == "area_gages2")[0]]
    meanprep = val_c[:, np.where(np.array(cfg.attrLst) == "p_mean")[0]]
    y_temp = basin_norm(train_y, basinarea, meanprep, True)

    series_data = np.concatenate([train_x, y_temp], axis=2)
    series_vars = cfg.varF + ["runoff"]

    # Fit scaler
    scaler = HydroScaler(
        attrLst=cfg.attrLst, seriesLst=series_vars,
        xNanFill=0.0, log_norm_cols=cfg.log_norm_cols,
    )
    attr_norm, series_norm = scaler.fit_transform(train_c, series_data)
    attr_norm[np.isnan(attr_norm)] = 0.0

    train_x = series_norm[:, :, :-1]
    train_x[np.isnan(train_x)] = 0.0
    train_y = np.expand_dims(series_norm[:, :, -1], 2)
    train_c = attr_norm

    val_c = scaler.transform(val_c, cfg.attrLst)
    val_x = scaler.transform(val_x, cfg.varF)
    val_c[np.isnan(val_c)] = 0.0
    val_x[np.isnan(val_x)] = 0.0

    print(f"Data loaded: train_x={train_x.shape} train_y={train_y.shape} train_c={train_c.shape}")

    return {
        "train_x": train_x, "train_y": train_y, "train_c": train_c,
        "val_x": val_x, "val_y": val_y, "val_c": val_c,
        "scaler": scaler, "basinarea": basinarea, "meanprep": meanprep,
        "gauge_ids": gauge_ids, "opt_data": opt_data,
    }

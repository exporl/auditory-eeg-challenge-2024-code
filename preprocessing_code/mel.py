"""Default save class."""
import gc
import json
import os
from typing import Any, Dict

import numpy as np

from brain_pipe.pipeline.cache.base import (
    pickle_dump_wrapper,
    pickle_load_wrapper,
)
from brain_pipe.save.base import Save
from brain_pipe.utils.multiprocess import MultiprocessingSingleton


def default_metadata_key_fn(data_dict: Dict[str, Any]) -> str:
    """Generate a key for the metadata.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.

    Returns
    -------
    str
        The key for the metadata.
    """
    if "data_path" in data_dict:
        return os.path.basename(data_dict["data_path"])

    if "stimulus_path" in data_dict and data_dict["stimulus_path"] is not None:
        return os.path.basename(data_dict["stimulus_path"])

    if "trigger_path" in data_dict and data_dict["trigger_path"] is not None:
        return os.path.basename(data_dict["trigger_path"])

    raise ValueError("No data_path or stimulus_path in data_dict.")


def default_filename_fn(data_dict, feature_name, set_name=None, separator="_-_"):
    """Generate a filename for the data_dict.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: Optional[str]
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.
    separator: str
        The separator to use between the different parts of the filename.

    Returns
    -------
    str
        The filename.
    """
    parts = []
    if "data_path" in data_dict:
        parts += [os.path.basename(data_dict["data_path"]).split(".")[0]]

    if "stimulus_path" in data_dict:
        parts += [os.path.basename(data_dict["stimulus_path"]).split(".")[0]]

    if feature_name is None and set_name is None:
        return separator.join(parts) + ".data_dict"

    if "event_info" in data_dict and "snr" in data_dict["event_info"]:
        parts += [str(data_dict["event_info"]["snr"])]

    keys = parts + [feature_name]
    if set_name is not None:
        keys = [set_name] + keys
    return separator.join(keys) + ".npy"


class DefaultSave(Save):
    """Default save class.

    This class will save data_dicts to disk, but also keep a metadata file
    (:attr:`Save.metadata_filename`) that contains the information about the mapping
    between an unprocessed input filename and multiple possible output filenames.
    """

    lock = MultiprocessingSingleton.manager.Lock()

    DEFAULT_SAVE_FUNCTIONS = {
        "npy": np.save,
        "pickle": pickle_dump_wrapper,
        "data_dict": pickle_dump_wrapper,
    }

    DEFAULT_RELOAD_FUNCTIONS = {
        "npy": np.load,
        "npz": np.load,
        "pickle": pickle_load_wrapper,
        "data_dict": pickle_load_wrapper,
    }

    def __init__(
        self,
        root_dir,
        to_save=None,
        overwrite=False,
        clear_output=False,
        filename_fn=default_filename_fn,
        save_fn=None,
        reload_fn=None,
        metadata_filename=".save_metadata.json",
        metadata_key_fn=default_metadata_key_fn,
    ):
        """Create a Save step.

        Parameters
        ----------
        root_dir: str
            The root directory where the data should be saved.
        to_save: Optional[Mapping[str, Any]]
            The data to save. If None, the data_dict is saved entirely. If a mapping
            between feature names and data is given, only the data for the given
            features is saved.
        overwrite: bool
            Whether to overwrite existing files.
        clear_output: bool
            Whether to clear the output data_dict after saving. This can save space
            when save is the last step in a pipeline.
        filename_fn: Callable[[Dict[str, Any], Optional[str], Optional[str], str], str]
            A function to generate a filename for the data. The function should take
            the data_dict, the feature name, the set name and a separator as input
            and return a filename.
        save_fn: Union[Callable[[Any, str], None], Mapping[str, Callable[[Any, str], None]], None]  # noqa: E501
            A function to save the data. The function should take the data and the
            filepath as inputs and save the data. If a mapping between file extensions
            and functions is given, the function corresponding to the file extension
            is used to save the data. If None, the default save functions (defined in
            self.DEFAULT_SAVE_FUNCTIONS) are used.
        reload_fn: Union[Callable[[str], Any], Mapping[str, Callable[[str], Any]], None]
            A function to reload the data. The function should take the filepath as
            input and return the data. If a mapping between file extensions and
            functions is given, the function corresponding to the file extension is
            used to reload the data. If None, the default reload functions (defined in
            self.DEFAULT_RELOAD_FUNCTIONS) are used.
        metadata_filename: str
            The filename of the metadata file.
        metadata_key_fn: Callable[[Dict[str, Any]], str]
            A function to generate a key for the metadata. The function should take
            the data_dict as input and return a key. This key will be used to check
            whether the data has already been saved.
        """
        super().__init__(clear_output=clear_output)
        self.root_dir = root_dir
        self.to_save = to_save
        self.filename_fn = filename_fn
        self.save_fn = save_fn
        if self.save_fn is None:
            self.save_fn = self.DEFAULT_SAVE_FUNCTIONS
        self.reload_fn = reload_fn
        if self.reload_fn is None:
            self.reload_fn = self.DEFAULT_RELOAD_FUNCTIONS
        self.metadata_filename = metadata_filename
        self.metadata_key_fn = metadata_key_fn
        self.overwrite = overwrite

    @property
    def overwrite(self):
        """Whether to overwrite existing files.

        Returns
        -------
        bool
            Whether to overwrite existing files.
        """
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        """Set whether to overwrite existing files.

        Parameters
        ----------
        value: bool
            Whether to overwrite existing files.
        """
        self._overwrite = value
        if self._overwrite:
            self._clear_metadata()

    def _single_obj_to_list(self, obj):
        if not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    def is_already_done(self, data_dict):
        """Check whether the data_dict has already been saved.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict to check.

        Returns
        -------
        bool
            Whether the data_dict has already been saved. This will be checked in the
            stored metadata.
        """
        if self.overwrite:
            return False
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        if key not in metadata:
            return False
        is_done = True
        found_filenames = self._single_obj_to_list(metadata[key])
        for filename in found_filenames:
            is_done = is_done and os.path.exists(os.path.join(self.root_dir, filename))
        return is_done

    def _clear_metadata(self):
        self.lock.acquire()
        metadata_path = os.path.join(self.root_dir, self.metadata_filename)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        self.lock.release()

    def _get_metadata(self):
        metadata_path = os.path.join(self.root_dir, self.metadata_filename)
        if not os.path.exists(metadata_path):
            return {}
        self.lock.acquire()
        with open(metadata_path) as fp:
            metadata = json.load(fp)
        self.lock.release()
        return metadata

    def _add_metadata(self, data_dict, filepath):
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        if key not in metadata:
            metadata[key] = []
        all_filepaths = self._single_obj_to_list(filepath)
        for path in all_filepaths:
            filename = os.path.relpath(path, self.root_dir)
            if filename not in metadata[key]:
                metadata[key] += [filename]
        self._write_metadata(metadata)

    def _write_metadata(self, metadata):
        metadata_path = os.path.join(self.root_dir, self.metadata_filename)
        self.lock.acquire()
        with open(metadata_path, "w") as fp:
            json.dump(metadata, fp)
        self.lock.release()

    def _serialization_wrapper(self, fn, filepath, *args, action="save", **kwargs):
        if not isinstance(fn, dict):
            return fn(filepath, *args, **kwargs)
        suffix = os.path.basename(filepath).split(".")[-1]
        if suffix not in fn:
            raise ValueError(
                f"Can't find an appropriate function to {action} '{filepath}'."
            )
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        return fn[suffix](filepath, *args, **kwargs)

    def _apply_to_data(self, data_dict, fn):
        if self.to_save is None:
            path = os.path.join(self.root_dir, self.filename_fn(data_dict, None, None))
            self._serialization_wrapper(fn, path, data_dict, action="save")
            self._add_metadata(data_dict, [path])
            return

        paths = []
        for feature_name, feature_loc in self.to_save.items():
            data = data_dict[feature_loc]
            if isinstance(data, dict):
                for set_name, set_data in data.items():
                    filename = self.filename_fn(data_dict, feature_name, set_name)
                    path = os.path.join(self.root_dir, filename)
                    self._serialization_wrapper(fn, path, set_data, action="save")
                    paths += [path]
            else:
                filename = self.filename_fn(data_dict, feature_name)
                path = os.path.join(self.root_dir, filename)
                self._serialization_wrapper(fn, path, data, action="save")
                paths += [path]
        self._add_metadata(data_dict, paths)

    def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
        """Check whether an already processed data_dict can be reloaded.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict for which we want to reload the already processed version.

        Returns
        -------
        bool
            Whether an already processed data_dict can be reloaded to continue
            processing.
        """
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        #print (key)
        print(f'key: {key}')
        print(f'metadata: {metadata}')
        if key not in metadata:
            return False
        print(f'the value of key is: {metadata[key]}')
        # TODO: implement reload for multiple files

        path = os.path.join(self.root_dir, metadata[key][0])
        # if len(metadata[key]) != 1:
        #     path = os.path.join(self.root_dir, metadata[key][-1])
        # print (path)
        print(f'path: {path}')
        print(f'overwrite: {self.overwrite}')
        if os.path.exists(path) and not self.overwrite:
            return True
        else:
            return False

    def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reload the data_dict from the saved file.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict for which we want to reload the already processed version.

        Returns
        -------
        Dict[str, Any]
            The reloaded data_dict.
        """
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        all_filepaths = self._single_obj_to_list(metadata[key])
        print(f'all_filepaths: {all_filepaths}')

        return self._serialization_wrapper(
            self.reload_fn,
            os.path.join(self.root_dir, all_filepaths[-1]),
            action="reload",
        )

    def __call__(self, data_dict):
        """Save the data_dict to the :attr:`root_dir`.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data_dict to save.

        Returns
        -------
        Optional[Dict[str, Any]]
            The data_dict if :attr:`clear_output` is False, None otherwise.
        """
        os.makedirs(self.root_dir, exist_ok=True)
        self._apply_to_data(data_dict, self.save_fn)
        # Save some RAM space
        if self.clear_output:
            # Explicitly clean up
            gc.collect()
            return None
        return data_dict

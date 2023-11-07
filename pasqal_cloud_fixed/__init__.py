import time
from typing import Any, Dict, List, Optional
from warnings import warn

from pasqal_cloud import SDK
from pasqal_cloud.batch import RESULT_POLLING_INTERVAL, Batch
from pasqal_cloud.device import EmulatorType
from pasqal_cloud.errors import *
from requests.exceptions import HTTPError

from .base_config import MyBaseConfig
from .batch import MyBatch
from .result_type import MyResultType


class MySDK(SDK):
    def create_batch(
        self,
        serialized_sequence: str,
        jobs: List[Dict[str, Any]],
        emulator: Optional[EmulatorType] = None,
        configuration: Optional[MyBaseConfig] = None,
        wait: bool = False,
        fetch_results: bool = False,
    ) -> MyBatch:
        """Create a new batch and send it to the API.
        For Iroise MVP, the batch must contain at least one job and will be declared as
        complete immediately.

        Args:
                serialized_sequence: Serialized pulser sequence.
                jobs: List of jobs to be added to the batch at creation.
                emulator: The type of emulator to use,
                  If set to None, the device_type will be set to the one
                  stored in the serialized sequence
                configuration: A dictionary with extra configuration for the emulators
                 that accept it.
                wait: Whether to wait for the batch to be done and fetch results
                fetch_results (deprecated): Whether to wait for the batch to
                  be done and fetch results


        Returns:
                Batch: The new batch that has been created in the database.

        Raises:
                BatchCreationError: If batch creation failed
                BatchFetchingError: If batch fetching failed
        """
        if fetch_results:
            warn(
                "Argument `fetch_results` is deprecated and will be removed in a"
                " future version. Please use argument `wait` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        req = {
            "sequence_builder": serialized_sequence,
            "webhook": self.webhook,
            "jobs": jobs,
        }

        # the emulator field is only added in the case
        # an emulator job is requested otherwise it's left empty
        if emulator:
            req.update({"emulator": emulator})

        # The configuration field is only added in the case
        # it's requested
        if configuration:
            req.update({"configuration": configuration.to_dict()})  # type: ignore

        try:
            batch_rsp = self._client._send_batch(req)
        except HTTPError as e:
            raise BatchCreationError(e) from e

        batch_id = batch_rsp["id"]
        if wait or fetch_results:
            while batch_rsp["status"] in ["PENDING", "RUNNING"]:
                time.sleep(RESULT_POLLING_INTERVAL)
                batch_rsp = self._get_batch(batch_id)

        batch = MyBatch(**batch_rsp, _client=self._client)

        self.batches[batch.id] = batch
        return batch

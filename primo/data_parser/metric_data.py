#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Standard libs
import logging
from typing import Callable, Dict, Optional

# Installed libs
import numpy as np
import pandas as pd
from pyomo.common.config import Bool, ConfigValue

# User-defined libs
from primo.data_parser.default_data import (
    SUPP_EFF_METRICS,
    SUPP_IMPACT_METRICS,
    _SupportedContent,
)
from primo.utils.config_utils import UserPriorities
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


class Metric:  # pylint: disable=too-many-instance-attributes
    """
    Dataclass for creating metric objects
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        weight: int,
        min_weight: int = 0,
        max_weight: int = 100,
        full_name: Optional[str] = None,
    ) -> None:
        """
        Returns a `Metric` object

        Parameters
        ----------
        name : str
            Short name of the metric.
            Must be a valid python variable name

        weight : int
            Weight associated with the metric

        min_weight : int, default = 0
            Lower bound on metric weight

        max_weight : int, default = 100
            Upper bound on metric weight

        full_name : str, default = None
            Elaborate name of the metric

        Raises
        ------
        ValueError
            If an invalid python variable name is provided,
            If `weight` lies outside the bounds,
            If an attempt is made to overwrite `name`
        """

        self.name = name
        self.full_name = name if full_name is None else full_name
        self.is_submetric = False
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight = weight

        # Is this a Yes/No or True/False type metric?
        self.is_binary_type = False
        # For some metrics, a higher value implies a lower priority for plugging.
        # Is this one of those metrics? E.g., compliance, production volume, etc.
        self.has_inverse_priority = False

        # Name of the column that contains the data
        # needed for calculating priority score
        self.data_col_name = None
        # Name of the column that contains the priority score
        self._score_col_name = None
        # Name of the project attribute that contains the efficiency score
        self._score_attribute = None

        # Value to fill with, if the data needed for the analysis of this metric
        # is not provided.
        self._fill_missing_value = ConfigValue(
            doc=(
                f"Value to fill with, if {self.full_name} information "
                f"is not provided"
            )
        )
        self._fill_missing_value._name = self.name

    def __str__(self) -> str:
        """Format for printing the object"""
        return (
            f"Metric name: {self.full_name}, Metric weight: {self.weight} \n"
            f"    Admissible range: [{self.min_weight}, {self.max_weight}]"
        )

    @property
    def name(self):
        """Getter for attribute `name`"""
        return self._name

    @name.setter
    def name(self, val: str):
        """Setter for attribute `name`"""
        if hasattr(self, "_name"):
            raise_exception(
                "Metric's key name cannot be modified after it is defined.",
                ValueError,
            )

        if not val.isidentifier():
            msg = (
                f"Received {val} for Metric's key name, "
                f"which is not a valid python variable name!"
            )
            raise_exception(msg, ValueError)

        self._name = val

    @property
    def weight(self):
        """Getter for attribute `weight`"""
        return self._weight

    @weight.setter
    def weight(self, val: int):
        """Setter for attribute `weight`"""
        if not isinstance(val, int):
            LOGGER.warning(
                f"Received {val}, a non-integer value for weight. "
                f"Rounding it to {round(val)}, the nearest integer value."
            )
            val = round(val)

        if self.min_weight <= val <= self.max_weight:
            self._weight = val
            return

        msg = (
            f"Attempted to assign {val} for metric {self.name}, which "
            f"lies outside the admissible range "
            f"[{self.min_weight}, {self.max_weight}]."
        )
        raise_exception(msg, ValueError)

    @property
    def effective_weight(self):
        """Getter for attribute `effective_weight`"""
        # NOTE: For primary metrics, effective_weight = weight. But for submetrics,
        # effective_weight = (weight / 100) * parent_metric.weight
        return self._weight

    @property
    def score_col_name(self):
        """Getter for attribute `score_col_name`"""
        if self._score_col_name is not None or self.data_col_name is None:
            return self._score_col_name

        self._score_col_name = (
            self.data_col_name + f" Score [0-{self.effective_weight}]"
        )
        return self._score_col_name

    @property
    def score_attribute(self):
        """
        Getter for attribute 'score_col_attribute' this is for attributes on the project class
        """
        if self._score_attribute is not None or self.data_col_name is None:
            return self._score_attribute
        self._score_attribute = self.name + f"_eff_score_0_{self.effective_weight}"
        return self._score_attribute

    @property
    def fill_missing_value(self):
        """
        Returns the value to fill with, if the data needed for the analysis of this
        metric is missing for a well in the dataset.
        """
        return self._fill_missing_value.value()

    @fill_missing_value.setter
    def fill_missing_value(self, value):
        """Setter for the _fill_missing_value attribute."""
        self._fill_missing_value.set_value(value)

    def _configure_fill_missing_value(self, domain: Callable, default=None):
        """
        Sets the domain validator and default value for `fill_missing_value` for
        supported metrics/submetrics

        Parameters
        ----------
        domain : function
            A valid domain validator function

        default : Any value satisfying the domain, default = None
            Default value for `fill_missing_value`
        """
        # If a domain already exists, and it is being updated, and if the existing non-None
        # value lies outside the domain, then ConfigValue throws an error.
        # To avoid this error, first set the value to None before updating the domain
        self._fill_missing_value.set_value(None)
        self._fill_missing_value.set_domain(domain)
        self._fill_missing_value.set_value(default)

        # Set the metric type to be binary if the domain is bool
        if domain is bool or domain is Bool:
            self.is_binary_type = True


class SubMetric(Metric):
    """
    Dataclass for creating submetric objects
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        parent_metric: Metric,
        weight: int,
        min_weight: int = 0,
        max_weight: int = 100,
        full_name: Optional[str] = None,
    ) -> None:
        """
        Returns a `SubMetric` object

        Parameters
        ----------
        name : str
            Short name of the metric.
            Must be a valid python variable name

        parent_metric : Metric
            Parent metric object

        weight : int
            Weight associated with the metric

        min_weight : int, default = 0
            Lower bound on metric weight

        max_weight : int, default = 100
            Upper bound on metric weight

        full_name : str, default = None
            Elaborate name of the metric

        Raises
        ------
        ValueError
            If an invalid python variable name is provided,
            If `weight` lies outside the bounds,
            If an attempt is made to overwrite `name`
        """

        super().__init__(name, weight, min_weight, max_weight, full_name)
        self.is_submetric = True
        self.parent_metric = parent_metric

    def __str__(self) -> str:
        """Format for printing objects"""
        return (
            f"Submetric name: {self.full_name}, Submetric weight: {self.weight} \n"
            f"    Admissible range: [{self.min_weight}, {self.max_weight}] \n"
            f"    Is a submetric of {self.parent_metric.full_name}"
        )

    @property
    def effective_weight(self):
        """Getter for attribute `effective_weight`"""
        return (self.weight / 100) * self.parent_metric.weight


class SetOfMetrics:
    """Container for storing metrics and submetrics"""

    def __init__(
        self, supported_metrics: Optional[Dict[str, _SupportedContent]] = None
    ):
        if supported_metrics is not None:
            for val in supported_metrics.values():
                if not val.is_submetric:
                    self.register_new_metric(
                        name=val.name,
                        full_name=val.full_name,
                    )

                else:
                    self.register_new_submetric(
                        name=val.name,
                        parent_metric=getattr(self, val.parent_metric),
                        full_name=val.full_name,
                    )
                metric = getattr(self, val.name)
                metric._required_data = val.required_data
                metric.has_inverse_priority = val.has_inverse_priority
                if val.fill_missing_value is not None:
                    metric._configure_fill_missing_value(**val.fill_missing_value)

    def __iter__(self):
        # Makes the object iterable! Iterate over all Metric and SubMetric objects.
        return iter(self.__dict__.values())

    def __contains__(self, var):
        return var in self.__dict__ or var in [obj.full_name for obj in self]

    def __setattr__(self, name: str, value: Metric):
        if not isinstance(value, Metric):
            msg = (
                f"Attributes of {self.__class__.__name__} must be instances of Metric. "
                f"Attempted to register {value}."
            )
            raise_exception(msg, TypeError)

        if name in self:
            msg = (
                f"Metric/submetric {name} has already been registered. "
                f"Attempting to register a new metric with the same name."
            )
            raise_exception(msg, ValueError)

        return super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Nicely formats the output when the object is printed.
        """
        _index = []
        data = {"Metric Name": [], "Metric weight": []}

        for obj in self.get_primary_metrics.values():
            _index.append(obj.name)
            data["Metric Name"].append(obj.full_name)
            data["Metric weight"].append(obj.weight)

        _index.append("")
        data["Metric Name"].append("Total")
        data["Metric weight"].append(sum(data["Metric weight"]))

        output = str(pd.DataFrame(data, columns=list(data.keys()), index=_index))

        for obj in self.get_primary_metrics.values():
            if not hasattr(obj, "submetrics"):
                continue

            output += f"\n\n\nPrimary metric {obj.full_name}, with weight {obj.weight},"
            output += " has submetrics:\n" + ("=" * 80) + "\n"
            _index = []
            data = {"Submetric Name": [], "Submetric weight": []}

            for sub_obj in obj.submetrics.values():
                _index.append(sub_obj.name)
                data["Submetric Name"].append(sub_obj.full_name)
                data["Submetric weight"].append(sub_obj.weight)

            _index.append("")
            data["Submetric Name"].append("Total")
            data["Submetric weight"].append(sum(data["Submetric weight"]))

            output += str(pd.DataFrame(data, columns=list(data.keys()), index=_index))

        return output

    def items(self):
        """
        Retrieves metric names and the associated metric objects
        """
        return self.__dict__.items()

    @property
    def get_primary_metrics(self):
        """Getter for all primary metrics"""
        return {key: obj for key, obj in self.items() if not obj.is_submetric}

    @property
    def get_submetrics(self):
        """
        Getter for all submetrics
        Returns a dict of dicts: {parent_metric_1 : {submetric_1 : obj_1,...}, ...}
        """
        return {
            key: val.submetrics
            for key, val in self.get_primary_metrics.items()
            if hasattr(val, "submetrics")
        }

    @property
    def _get_all_metrics_extended(self):
        """Getter for metric objects with both `name` and `full_name` as keys"""
        _extended_metrics = {obj.name: obj for obj in self}
        _extended_metrics.update({obj.full_name: obj for obj in self})

        return _extended_metrics

    def register_new_metric(
        self, name: str, weight: int = 0, full_name: Optional[str] = None
    ):
        """
        Registers a new metric

        Parameters
        ----------
        name : str
            Metric name, must be a valid python variable name

        weight : int
            weight associated with the metric

        full_name : str
            Elaborate name of the metric
        """
        setattr(self, name, Metric(name=name, weight=weight, full_name=full_name))

    def delete_metric(self, name: str):
        """
        Deletes an existing metric

        Parameters
        ----------
        name : str
            Metric `name`/`full_name`
        """
        _extended_metrics = self._get_all_metrics_extended
        obj = _extended_metrics.get(name, None)

        # Raise an error if the metric does not exist
        if obj is None:
            raise_exception(f"Metric/submetric {name} does not exist.", AttributeError)

        # If submetrics exist, delete all submetrics too
        sub_obj_list = [*obj.submetrics.values()] if hasattr(obj, "submetrics") else []

        for val in sub_obj_list:
            self.delete_submetric(val.name)

        delattr(self, obj.name)

    def register_new_submetric(
        self,
        name: str,
        parent_metric: Metric,
        weight: int = 0,
        full_name: Optional[str] = None,
    ):
        """
        Registers a new submetric

        Parameters
        ----------
        name : str
            Submetric name, must be a valid python variable name

        parent_name : Metric
            Parent metric object

        weight : int
            weight associated with the metric

        full_name : str
            Elaborate name of the metric
        """
        setattr(
            self,
            name,
            SubMetric(
                name=name,
                parent_metric=parent_metric,
                weight=weight,
                full_name=full_name,
            ),
        )
        obj = getattr(self, name)

        # Register the submetric in the parent metric object for convenience
        if hasattr(parent_metric, "submetrics"):
            parent_metric.submetrics[obj.name] = obj
        else:
            parent_metric.submetrics = {obj.name: obj}

    def delete_submetric(self, name: str):
        """
        Deletes an existing submetric

        Parameters
        ----------
        name : str
            Submetric `name`/`full_name`
        """
        _extended_metrics = self._get_all_metrics_extended
        obj = _extended_metrics.get(name, None)

        # Raise an error if the metric does not exist
        if obj is None:
            raise_exception(f"Submetric {name} does not exist.", AttributeError)

        parent_obj = obj.parent_metric
        parent_obj.submetrics.pop(obj.name)
        delattr(self, obj.name)

        # If all submetrics have been removed, then delete the attribute
        if len(parent_obj.submetrics) == 0:
            delattr(parent_obj, "submetrics")

    def set_weight(
        self,
        primary_metrics: Dict[str, int],
        submetrics: Optional[Dict[str, Dict[str, int]]] = None,
        check_validity: bool = True,
    ):
        """
        Sets/updates the weights of all metrics/submetrics.

        Parameters
        ----------
        primary_metrics : dict
            Dictionary containing the weights of primary metrics.
            key must be the `name` of the primary metric and the value
            must be its weight. Submetrics can also be included in this
            dictionary.

        submetrics : dict(dict), default = None
            Dictionary of dictionaries to update the weights of submetrics.
            This is optional, and it is useful to avoid confusion. The data must
            be of the form
            {primary_metric_name: {"submetric_1": weight_1, "submetric_2": weight_2}}

        check_validity : bool, default=True
            Checks if the weights are valid i.e., add up to 100
        """
        # Having a mutable object as a default is not recommended.
        # Apparently, this is the recommended way to set a mutable object
        # as the default.
        if submetrics is None:
            submetrics = {}

        _extended_metrics = self._get_all_metrics_extended

        # Add submetrics to primary metrics
        for key in submetrics:
            primary_metrics.update(submetrics[key])

        for key, obj in _extended_metrics.items():
            if key in primary_metrics:
                obj.weight = primary_metrics.pop(key)

        # If there are any unused elements, raise an error.
        if len(primary_metrics) > 0:
            raise_exception(
                f"Metrics/submetrics {[*primary_metrics]} are not recognized/registered.",
                KeyError,
            )

        # Check validity of the data
        if check_validity:
            self.check_validity()

    def check_validity(self):
        """
        Checks the validity of the weight values.
        Returns None if the weights are valid.
        Raises ValueError if the weights are not valid.
        """
        primary_metric_sum = sum(
            obj.weight for obj in self.get_primary_metrics.values()
        )

        if not np.isclose(primary_metric_sum, 100):
            raise_exception(
                "Sum of weights of primary metrics does not add up to 100", ValueError
            )

        for key, val in self.get_submetrics.items():
            sub_metric_sum = sum(sub_val.weight for sub_val in val.values())

            parent_metric = getattr(self, key)
            # If parent metric is inactive, then sum must be zero
            if parent_metric.weight == 0 and not np.isclose(sub_metric_sum, 0):
                msg = (
                    f"Weight of the primary metric {key} is zero, but the sum of "
                    f"weights of its submetrics is {sub_metric_sum}, which is nonzero."
                )
                raise_exception(msg, ValueError)

            if parent_metric.weight > 0 and not np.isclose(sub_metric_sum, 100):
                msg = (
                    f"Sum of weights of submetrics of the primary metric {key} "
                    f"does not add up to 100."
                )
                raise_exception(msg, ValueError)

    def build_widget(self, increments: int = 1):
        """
        Builds a widget to for visualizing and updating metrics

        Parameters
        ----------
        increments : int, default=5
            Increment value for the slider
        """
        counter = 1
        _to_widget_labels = {}
        _from_widget_labels = {}

        widget_data = {}
        for obj in self.get_primary_metrics.values():
            key = f"{counter}. {obj.full_name}"
            _to_widget_labels[obj.name] = key
            _from_widget_labels[key] = obj

            widget_data[key] = {
                "default": obj.weight,
                "min_val": obj.min_weight,
                "max_val": obj.max_weight,
                "incr": increments,
            }

            if not hasattr(obj, "submetrics"):
                counter += 1
                continue

            widget_data[key]["sub_weights"] = {}
            sub_counter = 1

            for sub_obj in obj.submetrics.values():
                sub_key = f"{counter}.{sub_counter} {sub_obj.full_name}"
                _to_widget_labels[sub_obj.name] = sub_key
                _from_widget_labels[sub_key] = sub_obj

                widget_data[key]["sub_weights"][sub_key] = {
                    "default": sub_obj.weight,
                    "min_val": sub_obj.min_weight,
                    "max_val": sub_obj.max_weight,
                    "incr": increments,
                }

                sub_counter += 1

            counter += 1

        _widget = UserPriorities(widget_data)
        _widget.set_widget_label_maps(_to_widget_labels, _from_widget_labels)
        return _widget

    def set_weight_from_widget(self, widget_obj: UserPriorities):
        """
        Updates the weight from the widget to the data object

        Parameters
        ----------
        widget_obj : UserPriorities
            Widget object containing the weights.
        """
        priority_weights, sub_priority_weights = widget_obj.return_value()

        # Add submetrics to priority weights
        for key in sub_priority_weights:
            priority_weights.update(sub_priority_weights[key])

        _, from_widget_labels = widget_obj.get_widget_label_maps()
        for key, obj in from_widget_labels.items():
            if key in priority_weights:
                obj.weight = priority_weights[key]
            else:
                obj.weight = 0

        self.check_validity()


class ImpactMetrics(SetOfMetrics):
    """Set of supported impact metrics"""

    def __init__(
        self,
        impact_metrics: Optional[Dict[str, _SupportedContent]] = None,
    ) -> None:
        # Default arguments retain values between function calls.
        # If we use a mutable argument, unexpected things can occurs.
        # See this for example:
        # https://stackoverflow.com/questions/26320899/
        if impact_metrics is None:
            impact_metrics = SUPP_IMPACT_METRICS
        super().__init__(impact_metrics)


class EfficiencyMetrics(SetOfMetrics):
    """Set of supported efficiency metrics"""

    def __init__(
        self,
        efficiency_metrics: Optional[Dict[str, _SupportedContent]] = None,
    ) -> None:
        if efficiency_metrics is None:
            efficiency_metrics = SUPP_EFF_METRICS
        super().__init__(efficiency_metrics)

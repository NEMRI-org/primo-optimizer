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
import copy
import inspect
import itertools
import logging
from typing import Dict, Union

# Installed libs
import numpy as np
import pyomo.environ as pyo
from gurobipy import GRB  # pylint: disable=no-name-in-module
from pyomo.environ import check_optimal_termination

# User-defined libs
from primo.data_parser.data_model import OptInputs
from primo.opt_model import FEASIBILITY_TOLERANCE
from primo.opt_model.base_model import BaseModel
from primo.utils import get_solver
from primo.utils.opt_utils import budget_slack_variable_scaling, is_pyomo_model_feasible
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable=super-init-not-called
class OptModel(BaseModel):
    """
    Class that provides the attributes and methods for the optimization model.
    """

    def __init__(
        self,
        model_name: str,
        model_inputs: OptInputs,
    ):
        """
        Initialize the OptModel class.

        Parameters
        ----------
        model_name : str
            A string identifier assigned to the optimization model.
        model_inputs : OptInputs
            The full inputs to the optimization model.
        """
        self.model_name = model_name
        self.model_inputs = model_inputs
        self.model = pyo.ConcreteModel()
        self.budget_sufficient = None
        self._threshold = model_inputs.distance_threshold
        self._max_wells_per_owner = model_inputs.max_wells_per_owner

    def _add_sets(self):
        """
        Define sets associated with the optimization model
        """
        LOGGER.info("Initializing set of wells!")
        model_inputs = self.model_inputs
        model = self.model
        model.s_w = pyo.Set(
            doc="Set of all wells", initialize=model_inputs.well_dict.keys()
        )

        disadvantaged_wells = [
            well_id
            for well_id, well in model_inputs.well_dict.items()
            if well.is_disadvantaged
        ]
        model.s_dw = pyo.Set(
            doc="Set of wells in disadvantaged communities",
            initialize=disadvantaged_wells,
        )
        model.s_cl = pyo.Set(
            doc="Set of all campaign candidates",
            initialize=model_inputs.campaign_candidates.keys(),
        )

        model.s_owc = pyo.Set(
            doc="Set of all owners",
            initialize=model_inputs.owner_well_count.keys(),
        )

        LOGGER.info("Initializing set of campaign candidates!")
        campaign_candidates = []
        for campaign_id, candidates in model_inputs.campaign_candidates.items():
            for well_id in candidates.well_dict:
                campaign_candidates.append((campaign_id, well_id))

        model.s_wcl = pyo.Set(
            doc="Set of all campaign candidates",
            initialize=campaign_candidates,
            dimen=2,
        )
        LOGGER.info("Finished initializing sets")

    def _add_parameters(self):
        """
        Define parameters associated with the optimization model
        """
        model_inputs = self.model_inputs
        model = self.model
        model.p_B = pyo.Param(
            doc="Total budget available for plugging",
            initialize=model_inputs.budget,
            mutable=False,
            within=pyo.PositiveReals,
        )

        objective_weights = {
            well_id: 0 if np.isnan(well.score) else well.score
            for well_id, well in model_inputs.well_dict.items()
        }
        model.p_v = pyo.Param(
            model.s_w,
            doc=("The metric score for each marginal well"),
            mutable=False,
            within=pyo.NonNegativeReals,
            initialize=objective_weights,
        )

        model.p_c = pyo.Param(
            pyo.NonNegativeIntegers,
            doc="Cost for plugging wells in a campaign",
            mutable=False,
            within=pyo.NonNegativeReals,
            initialize=model_inputs.mobilization_cost,
        )

        model.p_owc = pyo.Param(
            doc="Max wells per owner",
            initialize=model_inputs.max_wells_per_owner,
            mutable=False,
            within=pyo.Reals,
        )

        scaling_factor, self.budget_sufficient = budget_slack_variable_scaling(
            model_inputs, objective_weights
        )
        model.p_B_sl = pyo.Param(
            doc="Budget slack variable coefficient in the objective function",
            initialize=scaling_factor,
            mutable=False,
            within=pyo.PositiveReals,
        )

        LOGGER.info("Finished initializing parameters")

    def _add_variables(self):
        """
        Define variables associated with the optimization model
        """
        model = self.model
        model_inputs = self.model_inputs

        model.v_y = pyo.Var(model.s_w, within=pyo.Binary)

        campaign_set = []
        for campaign_id, campaign in model_inputs.campaign_candidates.items():
            for n_well in range(campaign.n_wells + 1):
                campaign_set.append((campaign_id, n_well))

        model.v_q = pyo.Var(campaign_set, within=pyo.Binary)
        model.v_B_slack = pyo.Var(within=pyo.NonNegativeReals)
        LOGGER.info("Finished adding variables")

    def _add_constraints(
        self,
        dac_budget_fraction: Union[float, None] = None,
        project_max_spend: Union[float, None] = None,
    ):
        """
        Define constraints associated with the optimization model

        Parameters
        ----------
        dac_budget_fraction : Union[float, None] = None
            if not None, assumes that the value provided is a percentage value (between 0 and 100)
            that denotes the amount of budget that must be used to plug wells in a DAC area and
            includes constraints to enforce the same in the opt problem

        project_max_spend : Union[float, None] = None
            if not None, assumes that the value provided is the maximum budget for a single project
            in dollars and includes constraints to enforce the same in the opt problem
        """
        model = self.model
        model_inputs = self.model_inputs

        def budget_constraint(model):
            """
            Define constraint that says we must not exceed the budget we have

            Parameters
            ----------
            model : pyo.ConcreteModel
                The optimization model.

            Returns
            -------
            pyo.Constraint
            """
            return (
                sum(
                    model.v_q[cluster, n_well] * model.p_c[n_well]
                    for (cluster, n_well) in model.v_q.index_set()
                )
                <= model.p_B
            )

        model.con_budget = pyo.Constraint(
            rule=budget_constraint, doc="Stay within budget"
        )

        LOGGER.info("Added budget constraints")

        def campaign_length(model, cluster_id):
            """
            Define constraints that help identify the length of the campaign.

            Parameters
            ----------
            model : pyo.ConcreteModel
                The optimization model
            cluster_id : int
                The identifier for the campaign cluster

            Returns
            -------
            pyo.Constraint
                The constraint ensuring the campaign length is correctly represented
            """
            campaign_candidates = model_inputs.campaign_candidates[cluster_id]
            wells = campaign_candidates.well_dict
            n_wells = len(wells)
            return (
                sum(model.v_y[w] for w in wells)
                - sum(
                    n_well * model.v_q[cluster_id, n_well]
                    for n_well in range(n_wells + 1)
                )
            ) == 0

        model.con_campaign_length = pyo.Constraint(
            model.s_cl, rule=campaign_length, doc="Define campaign length"
        )

        LOGGER.info("Added campaign length constraints")

        def campaign_length_symmetry_breaking(model, cluster_id):
            """
            Define constraints that break symmetry in campaign length constraints

            Parameters
            ----------
            model : pyo.ConcreteModel
                The optimization model.
            cluster_id : int
                The identifier for the campaign cluster.

            Returns
            -------
            pyo.Constraint
                The constraint ensuring the campaign_length constraints symmetry
                on RHS is broken
            """
            campaign_candidates = model_inputs.campaign_candidates[cluster_id]
            wells = campaign_candidates.well_dict
            n_wells = len(wells)
            return (
                sum(model.v_q[cluster_id, n_well] for n_well in range(n_wells + 1)) <= 1
            )

        model.con_campaign_length_symmetry_break = pyo.Constraint(
            model.s_cl, rule=campaign_length_symmetry_breaking, doc="Symmetry breaking"
        )

        LOGGER.info("Added campaign length symmetry breaking constraints")

        if self._threshold is not None:

            def skip_too_distant_wells(
                model, campaign_id_1, well_1, campaign_id_2, well_2
            ):
                """
                Define constraints that help avoid campaigns with too many distant
                wells in them.

                Parameters
                ----------
                model : pyo.ConcreteModel
                    The optimization model.
                campaign_id_1 : int
                    The identifier for the first campaign.
                well_1 : int
                    The identifier for the first well.
                campaign_id_2 : int
                    The identifier for the second campaign.
                well_2 : int
                    The identifier for the second well.

                Returns
                -------
                pyo.Constraint or pyo.Constraint.Skip
                    The constraint avoiding distant wells in the same campaign or
                    pyo.Constraint.Skip if the constraint should be skipped.
                """
                if campaign_id_1 != campaign_id_2:
                    return pyo.Constraint.Skip

                if well_1 == well_2:
                    return pyo.Constraint.Skip

                well_1_obj = model_inputs.well_dict[well_1]
                well_2_obj = model_inputs.well_dict[well_2]
                distance = well_1_obj.haversine_distance(well_2_obj)

                if distance <= self._threshold:
                    return pyo.Constraint.Skip

                return model.v_y[well_1] + model.v_y[well_2] <= 1

            model.con_skip_too_distant_wells = pyo.Constraint(
                model.s_wcl,
                model.s_wcl,
                rule=skip_too_distant_wells,
                doc="Avoid wells too far apart in a campaign",
            )
            LOGGER.info("Added constraints skipping distant wells in a campaign")

        if dac_budget_fraction is not None:

            def balance_budgets_disadvantaged_community(model):
                """
                Ensure that an appropriate amount of funding is reserved for serving
                wells in disadvantaged communities

                Parameters
                ----------
                model : pyo.ConcreteModel
                    The optimization model.

                Returns
                -------
                pyo.Constraint
                """

                return (
                    sum(model.v_y[well] for well in model.s_dw)
                    - dac_budget_fraction
                    / 100
                    * sum(model.v_y[well] for well in model.s_w)
                ) >= 0

            if dac_budget_fraction < 0 or dac_budget_fraction > 100:
                raise_exception(
                    "Invalid parameter received for budget fraction", ValueError
                )
            model.con_balanced_budgets = pyo.Constraint(
                rule=balance_budgets_disadvantaged_community,
                doc=(
                    "Ensure budgets are spent fairly among wells"
                    "in disadvantaged communities"
                ),
            )

        if model.p_owc > 0:

            def owner_well_count_limit(model, owner_id):
                """
                Define constraints that limit the maximum number of wells an owner can posses.

                Parameters
                ----------
                model : pyo.ConcreteModel
                    The optimization model.
                owner_id : str
                    The identifier for the owner.

                Returns
                -------
                pyo.Constraint
                    The constraint ensuring the number of wells belong to a single owner does not
                    exceed the user-defined threshold.
                """

                return (
                    sum(
                        model.v_y[well]
                        for well in model_inputs.owner_well_count[owner_id]
                    )
                ) <= model.p_owc

            model.owner_well_count = pyo.Constraint(
                model.s_owc, rule=owner_well_count_limit, doc="well by owner limit"
            )

        if project_max_spend is not None:

            def project_max_spend_rule(model, campaign_id):
                """
                Ensure that no project exceeds the max spend limits specified
                Parameters
                ----------
                model : pyo.ConcreteModel
                    The optimization model.
                campaign_id : int
                    The identifier for the campaign cluster
                Returns
                -------
                pyo.Constraint
                """
                campaign_candidates = model_inputs.campaign_candidates[campaign_id]
                wells = campaign_candidates.well_dict
                n_wells = len(wells)
                return (
                    sum(
                        model.v_q[campaign_id, n_well] * model.p_c[n_well]
                        for n_well in range(n_wells + 1)
                    )
                    <= project_max_spend
                )

            model.con_project_max_spend = pyo.Constraint(
                model.s_cl,
                rule=project_max_spend_rule,
                doc="Define limit on max project spend",
            )
            LOGGER.info("Added max project spend constraint")

        def budget_constraint_slack(model):
            """
            Implements a constraint that calculates the un-utilized
            amount of the available budget.


            Parameters
            ----------
            model : pyo.ConcreteModel
                The optimization model.

            Returns
            -------
            pyo.Constraint
            """
            return (
                model.p_B
                - sum(
                    model.v_q[cluster, n_well] * model.p_c[n_well]
                    for (cluster, n_well) in model.v_q.index_set()
                )
                <= model.v_B_slack
            )

        model.con_budget_slack = pyo.Constraint(
            rule=budget_constraint_slack,
            doc="Calculate the un-utilized amount of budget",
        )
        LOGGER.info("Added a constraint to calculate the un-utilized amount of budget")

        if self.budget_sufficient is False:

            def min_budget_usage(model):
                """
                Implements a constraint to ensure at least 50% of the
                budget is used when the budget is sufficient to
                plug all wells.

                Parameters
                ----------
                model : pyo.ConcreteModel
                    The optimization model.

                Returns
                -------
                pyo.Constraint
                """
                return (
                    sum(
                        model.v_q[cluster, n_well] * model.p_c[n_well]
                        for (cluster, n_well) in model.v_q.index_set()
                    )
                    >= 0.5 * model.p_B
                )

            model.con_min_budget = pyo.Constraint(
                rule=min_budget_usage,
                doc="Enforces a minimum budget usage of 50%",
            )

        LOGGER.info(
            "Added a constraint to ensure that at least 50 percent of the \
            budget is used when the total budget is sufficient to plug all wells."
        )

        LOGGER.info("Finished adding constraints")

    def _set_objective(self):
        """
        Define the objective function
        """
        model = self.model

        def obj(model):
            """
            Calculate the objective function for the optimization model.

            Parameters
            ----------
            model : pyo.ConcreteModel
                The optimization model.

            Returns
            -------
            pyo.Expression :
                The expression representing the objective function.
            """
            return (
                sum(model.v_y[w] * model.p_v[w] for w in model.s_w)
                - model.p_B_sl * model.v_B_slack
            )

        model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

        LOGGER.info("Finished defining objective!")

    def build_model(
        self,
        dac_budget_fraction: Union[float, None] = None,
        project_max_spend: Union[float, None] = None,
    ) -> None:
        """
        Initializes a model to represent the optimization problem at hand

        Parameters:
        -----------
        dac_budget_fraction : Union[float, None] = None
            if not None, assumes that the value provided is a percentage value (between 0 and 100)
            that denotes the amount of budget that must be used to plug wells in a DAC area and
            includes constraints to enforce the same in the opt problem

        project_max_spend : Union[float, None] = None
            if not None, assumes that the value provided is the maximum budget for a single project
            in dollars and includes constraints to enforce the same in the opt problem
        """
        LOGGER.info("Building optimization model!")
        self._add_sets()
        self._add_parameters()
        self._add_variables()
        self._add_constraints(
            dac_budget_fraction=dac_budget_fraction,
            project_max_spend=project_max_spend,
        )
        LOGGER.info("Finished building optimization model!")
        self._set_objective()

    def solve_model(self, solver_params: Union[Dict, None] = None) -> bool:
        """
        Solve the optimization model that has previously been set up using
        build_model method in the BaseModel class.

        Parameters
        ----------
        solver_params : Dict, optional
            A dictionary specifying solver parameters and values for them to be used.
            A minimum of "solver" value needing to be set

        Returns
        -------
        bool
            True if a usable (feasible or optimal) solution is found
            after the solve; False otherwise
        """

        def distance_callback(_, cb_opt, cb_where):
            """
            Includes constraints that prevent too distant wells to be included in the same project
            via Lazy callbacks
            """
            if cb_where == GRB.Callback.MIPSOL:
                model_inputs = self.model_inputs
                model = self.model
                for candidates in model_inputs.campaign_candidates.values():
                    selected_wells = set()
                    for well in candidates.well_dict:
                        cb_opt.cbGetSolution(vars=[model.v_y[well]])
                        val = model.v_y[well].value
                        if val is not None and np.isclose(val, 1):
                            selected_wells.add(well)

                        for well_1, well_2 in itertools.combinations(selected_wells, 2):
                            well_1_obj = model_inputs.well_dict[well_1]
                            well_2_obj = model_inputs.well_dict[well_2]
                            distance = well_1_obj.haversine_distance(well_2_obj)
                            if distance >= self._threshold:
                                LOGGER.info("Adding lazy cut!")
                                cb_opt.cbLazy(
                                    self.model.cons.add(
                                        model.v_y[well_1] + model.v_y[well_2] <= 1
                                    )
                                )

        if solver_params is None:
            options = {}
        else:
            options = copy.deepcopy(solver_params)

        # Retain stdout logs by default
        stream_output = options.get("stream_output", True)

        # Remove temporary files by default
        keepfiles = options.pop("keepfiles", False)

        # callback options
        callback = options.pop("LazyConstraints", 0)

        # Get arguments for solver object
        get_solver_args = {}
        parameters = inspect.signature(get_solver).parameters
        for parameter_name, parameter in parameters.items():
            default_val = parameter.default
            get_solver_args[parameter_name] = options.pop(parameter_name, default_val)

        get_solver_args["solver_options"] = options

        solver_obj = get_solver(**get_solver_args)

        solver = get_solver_args.get("solver")
        if solver == "gurobi_persistent":

            solver_obj.set_instance(self.model)
            callback = False

            if callback:
                self.model.cons = pyo.ConstraintList()
                solver_obj.options["LazyConstraints"] = 1
                solver_obj.set_callback(distance_callback)

        results = solver_obj.solve(self.model, keepfiles=keepfiles, tee=stream_output)

        if check_optimal_termination(results):
            return True

        # If solution is not proven optimal, let's check for feasibility
        return is_pyomo_model_feasible(self.model, FEASIBILITY_TOLERANCE)

    def get_results(self):
        """
        Return the well-id's selected to be plugged
        """
        model_inputs = self.model_inputs
        model = self.model
        wells = set()
        for campaign_id, candidates in model_inputs.campaign_candidates.items():
            selected = False
            for well in candidates.well_dict:
                val = model.v_y[well].value
                if val is None:
                    continue
                if np.isclose(val, 1):
                    LOGGER.info(f"Selected well: {well} for plugging")
                    selected = True
                    wells.add(well)
            if selected:
                LOGGER.info((f"Cluster: {campaign_id} selected for plugging"))
        return wells

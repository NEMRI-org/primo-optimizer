Optimization Model for Efficient P&A Campaigns
==============================================

Overview
--------

Given a set of marginal wells, an overall budget, and a radius for what constitutes an efficient project, the optimization model returns candidates for high efficiency P&A projects.


.. _mathematical_notation:

Optimization Model Notation
---------------------------


.. list-table:: **Sets**
        :widths: 25 75
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\textcolor{blue}{w \in W}`
          - Set of all marginal wells under consideration
        * - :math:`\textcolor{blue}{p \in P}`
          - Set of all priorities under consideration
        * - :math:`\textcolor{blue}{k \in K}`
          - Set of all projects


.. list-table:: **Parameters**
        :widths: 25 75
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\textcolor{brown}{B}`
          - Total budget available for P&A operations
        * - :math:`\textcolor{brown}{R}`
          - The maximum allowable distance in miles between two wells selected in the same project
        * - :math:`\textcolor{brown}{PL}`
          - The cost of plugging a single well
        * - :math:`\textcolor{brown}{C_{max}}`
          - The maximum allowable number of wells in a project
        * - :math:`\textcolor{brown}{d_{w_1, w_2}}`
          - The distance in miles between wells :math:`\textcolor{blue}{w_1}` and :math:`\textcolor{blue}{w_2}`
        * - :math:`\textcolor{brown}{K_{max}}`
          - The maximum number of projects to attempt to build
        * - :math:`\textcolor{brown}{v^p_w}`
          - The metric score assigned to each priority for each well (normalized to 0---100)
        * - :math:`\textcolor{brown}{m^p}`
          - The relative weight assigned to each priority (sum to 100)
        * - :math:`\textcolor{brown}{mc^i}`
          - The mobilization cost for a project assuming the number of wells is :math:`\textcolor{blue}{i}`
        * - :math:`\textcolor{brown}{VS}`
          - The scaling factor for the slack variable :math:`\textcolor{red}{SLB}` associated with the minimum budget constraint


.. list-table:: **Binary Variables**
        :widths: 25 75
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\textcolor{red}{x_{wk}}`
          - Is well selected in a specific project?
        * - :math:`\textcolor{red}{y_k}`
          - Is project selected or active?
        * - :math:`\textcolor{red}{N_k^i}`
          - If 1, :math:`\textcolor{blue}i` wells selected for plugging in project :math:`\textcolor{blue}k`


.. list-table:: **Continuous Variables**
        :widths: 25 75
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\textcolor{red}{SLB}`
          - The slack variable representing the difference between the total budget and the actual budget used


.. _mathematical_program_formulation:

Optimization Model Formulation
------------------------------

The objective function maximizes utility derived through the P&A projects. The utility is computed
through a weighted sum of all the priorities under consideration weighted with their relative importance.
The slack variable :math:`\textcolor{red}{SLB}` is implemented to impose a penalty when the well-plugging budget
is not fully utilized, thereby encouraging the maximum use of the allocated budget.

Please note that all priorities :math:`\textcolor{brown}{v^p_w}` are assigned a score for each well
between 0---100. Additionally, the relative weights :math:`\textcolor{brown}{m^p}` specified for each priority
in the objective must sum to 100.



**Objective**

.. math::
  :label: eq:obj 

  \max \sum_{k \in K} \sum_{w \in W} \sum_{p \in P} \textcolor{brown}{m^p} \times \textcolor{brown}{v^p_w} \times \textcolor{red}{x_{wk}} - \textcolor{brown}{VS} \times \textcolor{red}{SLB}

Note that the objective :eq:`eq:obj` ensures maximum utility from plugging projects, but does not ensure that all projects have a 
maximum utility. In other words, there could be a large variance in the utility of the projects suggested due to the objective above, with some large projects mixed with some very
small projects. If more balanced projects are desired, they could be achieved through:

- Setting a lower bound on the number of wells or the normalized utility from a project
- Using a max/min formulation where the objective maximizes the minimum of utilities over all projects

Development of the above options will be undertaken if the results from the current formulation are unsatisfactory.


**Budget**

It is important to stay within the available budget. For every project, there are two associated costs:

- Plugging
- Mobilization

The cost of plugging is assumed as a constant per well. Additionally, it is assumed that mobilization costs
follow economies of scale, namely, that as the number of wells in a project increases, the per-well cost
of mobilization decreases as shown in equation :eq:`eq:mob`.

.. math::
        :label: eq:mob

        \sum_{w \in W} \sum_{k \in K}  \textcolor{brown}{PL} \times \textcolor{red}{x_{wk}}  +
        \sum_{k \in K} \sum_{i=1}^{C_{max}} \textcolor{brown}{mc^i} \times \textcolor{red}{N_k^i}  \leq B

**Number of wells in a project**

The constraint described by equation :eq:`eq:now` ensures that the total number of wells in a campaign is correctly computed.

.. math::
        :label: eq:now

        \sum_{w \in W} \textcolor{red}{x_{wk}} = \sum_{i=1}^{\textcolor{brown}{C_{max}}} i \times \textcolor{red}{N_k^i} \quad \forall k \in K

        \sum_{i=1}^{\textcolor{brown}{C_{max}}} \textcolor{red}{N_k^i} \leq 1 \quad \forall k \in K

**Exclusivity of wells in a project**

The constraint described by equation :eq:`eq:exc` ensures that a marginal well is included in one project at most.

.. math::
        :label: eq:exc

        \sum_{k \in K} \textcolor{red}{x_{wk}} \leq 1 \quad \forall w \in W


**Compactness of projects**

The constraint described by equation :eq:`eq:comp` ensures that only wells within a pre-specified radius are included in the same project.

.. math::
        :label: eq:comp

        \textcolor{red}{x_{w_1k}} +  \textcolor{red}{x_{w_2k}} \leq 1 \quad \forall k \in K, \forall w_1 \in W, \forall w_2 \in W, w_1 < w_2, \textcolor{brown}{d_{w_1, w_2}} \geq \textcolor{brown}{R}


**Symmetry breaking**

The set of constraints in this section is not strictly required for correctness of the model, but can help speed up the search for the optimal solution by the solver
by breaking symmetries in the mathematical model.

The constraints described by equation :eq:`eq:sym` ensure that the first :math:`k` projects in lexicographic ordering are utilized: 

.. math::
        :label: eq:sym

        \textcolor{red}{x_{wk}} \leq \textcolor{red}{y_k} \quad \forall k \in K, \forall w \in W

        \textcolor{red}{y_k} \geq \textcolor{red}{y_{k+1}}  \quad \forall k \in K - \{K_{max}\}


The constraint described by equation :eq:`eq:sym2` ensures that the largest projects in terms of size of wells are sorted in lexicographic ordering.

.. math::
      :label: eq:sym2 

      \sum_{w \in W} \textcolor{red}{x_{wk}} \geq \sum_{w \in W} \textcolor{red}{x_{w(k+1)}} \quad \forall k \in K - \{K_{max}\}


**Budget usage**

The constraint defined by equation :eq:`eq:minb` ensures that the remaining excess budget is accurately calculated.

.. math::
        :label: eq:minb

        B - (\sum_{w \in W} \sum_{k \in K}  \textcolor{brown}{PL} \times \textcolor{red}{x_{wk}}  +
        \sum_{k \in K} \sum_{i=1}^{C_{max}} \textcolor{brown}{mc^i} \times \textcolor{red}{N_k^i}) \leq \textcolor{red}{SLB}
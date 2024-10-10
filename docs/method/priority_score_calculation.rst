Priority Score Calculation
==========================

Typical Priority Factor
-----------------------
PRIMO allows users to customize the priority factors used for ranking the MCWs. Some typical priority factors 
include:

- Methane emissions proxies

    * Leak [Yes/No]
    * Compliance [Yes/No]
    * Violation [Yes/No]
    * Incident [Yes/No]
- Disadvantaged community impact

    * Justice40 federal DAC
    * State DAC
- Sensitive receptors

    * Schools
    * Hospitals
    * Buildings
    * State wetlands
    * Federal wetlands
- Production volumes

    * One-year production volumes
    * Five-year production volumes
- Well age

- Owner well count
  
- ...

Calculate Priority Scores of Wells
----------------------------------
Depending on the type of data associated with the priority factor, priority scores are calculated using two methods.

Value-based Priority Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^
For factors where the corresponding data are numerical (e.g., well age and well depth), scores are normalized to a 
scale of 0 to 100 using the min-and-max method based on the source data. This ensures that wells receive varying 
scores under value-based priority factors.

Since weights are assigned to each priority factor, the score that a well receives under the priority factor is 
calculated using one of the following two equations:

**Priority Score**

*Prefer larger values*

For factors where larger values indicate higher priority, such as desiring older wells in a project, the
priority score is calculated using the equation :eq:`eq:prio1`:

.. math::
  :label: eq:prio1
    
  \textcolor{brown}{v^p} = \textcolor{brown}{m^p} \times \frac {(\textcolor{brown}{a} - \textcolor{brown}{a_{min}})} {(\textcolor{brown}{a_{max}} - \textcolor{brown}{a_{min}})}


*Prefer smaller values*

For factors where smaller values indicate higher priority, such as desiring wells with low oil/gas production, the
priority score is calculated using the equation :eq:`eq:prio2`:

.. math::
  :label: eq:prio2

  \textcolor{brown}{v^p} = \textcolor{brown}{m^p} \times \frac {(\textcolor{brown}{a_{max}} - \textcolor{brown}{a})} {(\textcolor{brown}{a_{max}} - \textcolor{brown}{a_{min}})}

.. list-table:: **Parameters**
        :widths: 25 75
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\textcolor{brown}{a}`
          - The original data associated with the priority factor of a well
        * - :math:`\textcolor{brown}{a_{max}}`
          - The maximum original data associated with the priority factor among all wells
        * - :math:`\textcolor{brown}{a_{min}}`
          - The minimum original data associated with the priority factor among all wells  
        * - :math:`\textcolor{brown}{m^p}`
          - The relative weight assigned with the priority factor
        * - :math:`\textcolor{brown}{v^p}`
          - The metric score of the priority factor for a well


Yes/No-Based Priority Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For factors where the corresponding information is either a **yes** or **no**, indicating whether the well meets the 
qualification (e.g., leaks), wells receive either full score or a zero based on user specifications.

Similarly, the score that a well receives under the priority factor includes the weight assigned to the factor.


Total Priority Score
^^^^^^^^^^^^^^^^^^^^
The total priority score that a well receives is the sum of all scores received under all priority factors. 
Well candidates are ranked based on their total score.
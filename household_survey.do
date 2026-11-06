
gl root "C:\Users\aalvarado\OneDrive - International Monetary Fund (PRD)\GTM\deportation project\deportation_gtm\data\"
gl eneic "C:\Users\aalvarado\OneDrive - International Monetary Fund (PRD)\GTM\ENEIC\"


/*
local relevant_var "ANIO TRIMESTRE FACTOR OCUPADOS DESOCUPADOS PET PEA INACTIVOS P00A10 P05C16 P05C09B P05C08A P05C08B P06C01 NUM_HOGAR NUM_PERSONA P05C07A P05C07B"
import spss using "$eneic\2025T1.sav", clear
keep `relevant_var'
tempfile b2025T1
save `b2025T1'

import spss using "$eneic\2025T2.sav", clear
ta TRIMESTRE
drop if TRIMESTRE==2
keep `relevant_var'
tempfile b2025T2
save `b2025T2'

import spss using "$eneic\2025T3.sav", clear
keep `relevant_var'
tempfile b2025T3
save `b2025T3'


import spss using "$eneic\2025T4.sav", clear
keep `relevant_var'
tempfile b2025T4
save `b2025T4'

import spss using "$eneic\2026T1.sav", clear
keep `relevant_var'
tempfile b2026T1
save `b2026T1'


clear 
append using `b2026T1'
append using `b2025T4'
append using `b2025T3'
append using `b2025T2'
append using `b2025T1'
egen id=group(NUM_HOGAR NUM_PERSONA)

save "$root\survey.dta", replace

*/

use "$root\survey.dta", clear
gen salaried = inrange(P05C16, 1, 4)

* Map TRIMESTRE code to actual quarter number (2->1, 3->2, 4->3, 5->4, 6->1)
gen byte qtr = .
replace qtr = TRIMESTRE - 1 if inrange(TRIMESTRE, 2, 5)
replace qtr = 1             if TRIMESTRE == 6

* Quarterly date (Stata %tq format)
gen quarter = yq(ANIO, qtr)
format quarter %tq
label var quarter "Quarter (Stata quarterly date)"


gen formal = 0 if PEA == 1 // model combines informal workers and job seekers.
replace formal=. if  P05C16==6 | P05C16==8 // not employer
replace formal = 1 if (P05C08A==1 & salaried==1)


xtset id quarter 
egen qid = group(quarter)
xtset id qid

preserve
collapse (sum) formal PEA PET [pw=FACTOR], by(quarter)
gen formal_rate=formal/PEA
gen participation=PEA/PET
list quarter formal_rate participation
restore


gen tenure_m = P05C07A*12 + P05C07B

* status next quarter (missing if not observed in t+1)
gen formal_next  = F.formal
gen tenure_next  = F.tenure_m

* sample: formal at t, observed at t+1
gen sample = formal == 1 & !missing(formal_next)

* (1) model-consistent: leaves formal employment
gen sep_exit = (formal_next == 0) if sample

* (2) broad: also counts formal-to-formal job changes (tenure reset)
gen sep_broad = (formal_next == 0 | (formal_next == 1 & tenure_next < 3)) if sample

* weighted quarterly rates, by quarter pair and pooled
table qid if sample [pw=FACTOR], stat(mean sep_exit sep_broad) stat(count sep_exit)

summarize sep_exit [aw=FACTOR] if sample
local dq = r(mean)
display "Quarterly delta (exit): " %6.4f `dq'
display "Annual delta_F (exit):  " %6.4f 1 - (1 - `dq')^4
# emergency_room
The purpose when I wrote this module was to write a rough model that predicts wait times and deaths of patients, based on multiple parameters (eg number of doctors, time doctors are allowed to see each patient, patient severity)
This module utilizes queues for patients, and patients are seen based on severity (documentation for module functions explains this more thoroughly).
The functionality for this was written in a way that patients are only seen if their severity lies between the tails of our minimum and maximum severity,
as patients at the less severe end of the distribution do not need to be seen right away, and patients in the more severe tail may be beyond saving.

The two main functions of interest are checking_in_patients and hospital_simulation. Providing documentation below:

def hospital_simulation(mins_to_sim, docs, sppm, avg_pat_ph, threshold_SD, mu, SD, max_line_length, sev_coef, wait_coef, interact_coef):
    """
    

    Parameters
    ----------
    mins_to_sim : int
        how many minutes you would like to simulate for the emergency room.
    docs : int
        number of doctors to be present in the emergency room.
    sppm : int
        severity points per minute - abstract metric of how many "severity points" doctors should be seeing per minute.
        the higher this value, the worse care patients are receiving.
    avg_pat_ph : int or float
        average number of patients per hour the hospital expects to see.
    threshold_SD : int or float
        the number of standard deviations away from patient severity average (mu), beyond which patients will not be 
        seen immediately at the end of the queue, unless length of queue < max_line_length.
        this applies to relatively severe patients and relatively mild patients.
    mu : int or float
        average severity of patient expected. patient severity will be drawn from a normal distribution ~N(mu, SD).
    SD : int or float
        standard deviation of patient severity distribution.
    max_line_length : int
        if patient severity is in one of the tails, meaning
        mu - threshold_SD*SD < patient.severity < mu - threshold_SD*SD, then we see patient right away.
        but if patient severity is above or below those tail values, and our patient.queue.size() > max_line_length,
        then such patients get sent to the back of the queue.
        it is assumed that mild patients do not need immediate attention, while very severe patients are beyond saving
    sev_coef : float
        when predicting patient outcomes, we use three parameters: patient severity, wait time, and an interaction 
        between patient severity and wait time. The three coefficients we associate with these parameters should sum to one.
        sev_coef is the coefficient attached to the patient severity parameter in this model. 
        this coefficient would be data driven in real life, but here we just input coefficients that we intuit.
        in real life, these coefficients are not variable - they are parameters given to us by data.
    wait_coef : float
        when predicting patient outcomes, we use three parameters: patient severity, wait time, and an interaction 
        between patient severity and wait time. The three coefficients we associate with these parameters should sum to one.
        wait_coef is the coefficient attached to the wait time parameter in this model. 
        this coefficient would be data driven in real life, but here we just input coefficients that we intuit.
        in real life, these coefficients are not variable - they are parameters given to us by data..
    interact_coef : float
        when predicting patient outcomes, we use three parameters: patient severity, wait time, and an interaction 
        between patient severity and wait time. The three coefficients we associate with these parameters should sum to one.
        interact_coef is the coefficient attached to the patient severity and wait time interaction parameter in this model. 
        this coefficient would be data driven in real life, but here we just input coefficients that we intuit.
        in real life, these coefficients are not variable - they are parameters given to us by data..

    Returns
    -------
    wait_sev_death : list
        list of tuples, with number of tuples being equal to number of patients generated through simulation
        tuples are in form (patient wait time, patient severity, death prediction (bool)).
    patient_stats : tuple
        tuple containing descriptive statistics:
            (avg_sev, avg_wait, patients_seen, remaining_patients, total_patients, num_deaths, death_rate)

    """    

def checking_in_patients(doc_dict, patient_queue, threshold_SD, mu, SD, wait_time_severity, current_minute, max_line_length):
    #max_line_length
    
    """
    Parameters
    ----------
    doc_dict : Dict
        Dictionary with Doctor objects as values, with numeric keys.
    patient_queue : Queue
        Queue containing patients.
    threshold_SD : float or int
        This is the number of standard deviations from your specified mu of patient severity distribution 
        (using normal distribution ~N(mu, SD) that determines whether a patient gets seen immediately 
         when they get to the end of the queue. If their severity is between the tails, meaning
         mu - threshold_SD*SD < patient.severity < mu - threshold_SD*SD, then we see patient right away.
         But if patient severity is above or below those tail values, and our patient.queue.size() > max_line_length,
         then such patients get sent to the back of the queue.
         
         We do this because patients in the high severity tail are probably beyond saving, while patients
         in the low severity tail do not need immediate help
    mu : float or int
        mean of normal distribution from which patient severity will be drawn.
    SD : float or int
        standard deviation of normal distribution from which patient severity will be drawn.
    wait_time_severity : list
        list containing tuple of patient's wait times and severities.
    current_minute : int
        current simulated used to generate patient timestamp when they enter queue, as well
        as determine their wait time when they exit queue.
    max_line_length : int
        If patient severity is in one of the tails, meaning
        mu - threshold_SD*SD < patient.severity < mu - threshold_SD*SD, then we see patient right away.
        But if patient severity is above or below those tail values, and our patient.queue.size() > max_line_length,
        then such patients get sent to the back of the queue..

    Returns
    -------
    None.

    """

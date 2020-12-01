#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:41:28 2020

@author: brendanbirch

The purpose of this module is to simulate an emergency rooms, and to attempt a prediction of
patient outcomes / patient deaths during a pandemic.

"""
#%%
# EMERGENCY ROOM SIMULATION
import numpy as np
import random
import pandas as pd


#%%
class Queue:  
    def __init__(self):
        self.items = []
    
    def isEmpty(self):
        return self.items == []
    
    def enqueue(self, item): #assumes left side of list is the back of the line
        self.items.insert(0, item)
        
    def dequeue(self): #assumes right side of list is the front of the line
        return self.items.pop()
    
    def size(self):
        return len(self.items)
    
    def peek(self):
        return self.items[len(self.items) - 1]

#%%

class Doctor:
    def __init__(self, sppm): 
        """
        Parameters
        ----------
        sppm : float or int
            severity points per minute. this is an abstract idea of the number of severity points
            a doctor will be cycling through per minute.
            The higher this number, the lower quality of care a doctor is giving each patient

        Returns
        -------
        None.

        """
        self.patient_rate = sppm
        self.current_patient = None
        self.time_remaining = 0 #time required for a patient will be determined by a randomly chosen level of seriousness for patient
        
    def tick(self):
        if self.current_patient != None:
            self.time_remaining -= 1
            if self.time_remaining <= 0:
                self.current_patient = None
    
    def busy(self):
        if self.current_patient != None:
            return True
        else:
            return False
        
    def meet_patient(self, new_patient):
        self.current_patient = new_patient
        self.time_remaining = new_patient.get_severity_points() / self.patient_rate
        #the amount of time a doctor spends with a patient is not constant
        #it is determined by the patient's severity, and the doctor's sppm allowance

#%%  
class Patient:
    def __init__(self, time, mu, SD):
        self.severity = np.random.normal(mu, SD) #average patient will have severity of mu, dist will have SD = SD
        self.timestamp = time
        
    def get_stamp(self):
        return self.timestamp
    
    def get_severity_points(self):
        return self.severity
    
    def wait_time(self, current_time):
        return current_time - self.timestamp

#%%
### get rid of the coefficient parameters. instead, have it spit out a prediction for each patient, based on a model you made beforehand

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
    doc_dict = {}
    for i in range(docs):
        doc_dict[i] = Doctor(sppm) #we're entering doctors as values in our dictionary
    
    patient_queue = Queue()
    wait_time_severity = []
    
    for current_minute in range(mins_to_sim):
        current_minute = current_minute
        
        num_new_patients = round(np.random.normal(avg_pat_ph/60, 3)) #assuming SD of 3 - all of this would actually be data driven
        for _ in range(num_new_patients):
            patient = Patient(current_minute, mu, SD)
            patient_queue.enqueue(patient)
                #our threshold is defined as the severity above which our patient will definitely die, so 
        checking_in_patients(doc_dict = doc_dict, 
                             patient_queue = patient_queue, 
                             threshold_SD = threshold_SD, 
                             mu = mu, 
                             SD = SD, 
                             wait_time_severity = wait_time_severity, 
                             current_minute = current_minute, 
                             max_line_length = max_line_length)
    
    remaining_patients = patient_queue.size()
    patients_seen = len(wait_time_severity)
    
    #the following two functionalities currently give division by zero after running checking_in_patients, so commenting them out
    #average_wait = sum([_[0] for _ in wait_time_severity]) / len(wait_time_severity)
    
    #print('average wait {:5.2f} mins, {} patients remaining'.format(average_wait, patient_queue.size()))
    
    if patient_queue.size() > 0: #here we're dequeuing patients still left at the end
        for patient in range(patient_queue.size()):
            next_patient = patient_queue.dequeue()
            wait_time_severity.append((next_patient.wait_time(current_minute), next_patient.severity))
    
    total_patients = len(wait_time_severity)
    times = [_[0] for _ in wait_time_severity]
    severities = [_[1] for _ in wait_time_severity]
    max_wait = max(times)
    max_severity = max(severities)
    avg_sev = np.mean(severities)
    avg_wait = np.mean(times)
    
    deaths = [np.random.binomial(n=1, p = sev_coef*sw[1]/max_severity + 
                                 wait_coef*sw[0]/max_wait + 
                                 interact_coef*sppm/(1+sppm) * sev_coef*sw[1]/max_severity * wait_coef*sw[0]/max_wait) for sw in wait_time_severity]
    
    num_deaths = sum(deaths)
    death_rate = num_deaths/total_patients
    #this is the most complex part of the model. it creates an array of deaths, which 
    #are bools, 1 representing a patient death.
    #each sw is a tuple in the wait_time_severity list
    #the coefficients for these variables are determined by intuition, but would be data driven if we were to really implement the model
    
    wait_sev_death = list(zip(times, severities, deaths))
    
    patient_stats = (avg_sev, avg_wait, patients_seen, remaining_patients, total_patients, num_deaths, death_rate)
    #this returns a list of (wait_time, severity, death_bool) for each patient
    
    return (wait_sev_death, patient_stats)

#what we need to do is model how many patients show up at any given minute,
#and for each one of them, create a new patient

#%%

def checking_in_patients(doc_dict, patient_queue, threshold_SD, mu, SD, wait_time_severity, current_minute, max_line_length):
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
    for _, doctor in doc_dict.items():
        if patient_queue.size() > 0:
            if mu - threshold_SD*SD <= patient_queue.peek().severity <= mu + threshold_SD*SD: 
                #we use this approach to determine whether a patient is in the tails of our specified normal distrubtion
                #with mu = mu, SD = SD. If a patient is really bad or really "good", then they get sent
                #to the back of the line if our line is longer than our predetermined max_line length allowance
                
                if (not doctor.busy() and not patient_queue.isEmpty()):
                    next_patient = patient_queue.dequeue()
                    wait_time_severity.append((next_patient.wait_time(current_minute), next_patient.severity))
                    #this is a tuple of (waiting time, patient severity), which we'll use to forecast whether a patient dies
                    doctor.meet_patient(next_patient)
            else:
                if patient_queue.size() < max_line_length:
                    #this is where interaction of line length and severity threshold comes in
                    #if patient is in one of our tails, then they get added if queue length is shorter than our specified max length
                    next_patient = patient_queue.dequeue()
                    wait_time_severity.append((next_patient.wait_time(current_minute), next_patient.severity))
                    #this is a tuple of (waiting time, patient severity), which we'll use to forecast whether a patient dies
                    doctor.meet_patient(next_patient)
                else:
                    next_patient = patient_queue.dequeue()
                    original_time = next_patient.timestamp #this allows us to keep track of original time patient arrived
                    sent_back_patient = Patient(original_time, mu, SD) 
                    #this creates a 'new' patient who is really the original patient
                    #instantiated with their original line entry time
                    patient_queue.enqueue(sent_back_patient)
                    
                    checking_in_patients(doc_dict, patient_queue, threshold_SD, mu, SD, wait_time_severity, current_minute, max_line_length)
                    '''
                    note here: if your line is entirely composed of tail-severity patients, then you will get stuck 
                    in this recursion. 
                        Hence, you should make a counter that tracks line length, and if counter > line_length,
                        then you need to start admitting patients.
                    '''
                    
                    #this is where we enter our recursion, so that we don't forget to assign a patient to this doctor
                    #the recursion will have us go back to the front of the new queue and repeat this process the
                    #new patient at the exit of the queue
                    
            doctor.tick()
        
        else:
            if (not doctor.busy() and not patient_queue.isEmpty()):
                    next_patient = patient_queue.dequeue()
                    wait_time_severity.append((next_patient.wait_time(current_minute), next_patient.severity))
                    #this is a tuple of (waiting time, patient severity), which we'll use to forecast whether a patient dies
                    doctor.meet_patient(next_patient)
            
            doctor.tick()
            

#%%

if __name__ == '__main__':
    waits = hospital_simulation(mins_to_sim = 1440, docs = 2, sppm = 5, 
                                avg_pat_ph = 30, sev_coef = .6, wait_coef = .2, 
                                interact_coef = .2, threshold_SD = 2, mu = 10, SD = 3, max_line_length = 3)

    print('the output order in this tuple is: avg_sev, avg_wait, patients_seen, remaining_patients, total_patients, num_deaths, death_rate')
    print(waits[1])
    # note that if you want to access the list of [(wait_times, severities, deaths)]
    # for each simulation, then simply print(waits[0])

    waits = hospital_simulation(mins_to_sim = 1440, docs = 10, sppm = 5, avg_pat_ph = 30, 
                                sev_coef = .6, wait_coef = .2, interact_coef = .2, 
                                threshold_SD = 2, mu = 10, SD = 3, max_line_length = 3)

    #here we see that we actually get a big change in death rate by just increasing doctors
    print(waits[1])





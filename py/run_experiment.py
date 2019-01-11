import binaural_recording
import perception_test
import plotting
import os
os.environ["SUBJECT"] = "test" # <-- Enter Subject here
os.environ["EXPDIR"] = "C:/Projects/MEG_Elevation/"


# STEP 1: localization test under free field conditions
response = perception_test.free_field()

#STEP 2: plot response --> Errors for all speakers must be samller than 1 speaker distance!
plotting.plot_response(response)

#STEP 3: record stimuli
binaural_recording.recording()

#STEP 4: plot recordings, the signal from right should be bigger than the signal from the left. spectra should look
#similar with some variation across speakers and from left to right.
plotting.plot_recordings()

# STEP 4: headphone test without adapter
response = perception_test.headphones()

#STEP 5: plot response --> Errors for all speakers must be samller than 1 speaker distance!
plotting.plot_response(response)

#STEP 6: headphone test with adapter
response = perception_test.headphones(adapter=True)

#STEP 7: plot response --> Errors for all speakers must be samller than 1 speaker distance!
plotting.plot_response(response)





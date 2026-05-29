## Questions
* For sensor_noise_attack.py how do we compare it's results from our random baseline? they're both random so it's kind of hard to compare. Unless a trained policy was needed in the first place? 
    * What policy would we use to train?

## Scenarios
### Scenario 4: Privacy in Robotics
Cybersecurity idea: Robots may collect sensitive data through cameras, sensors, locations, or user behavior logs.

Robotics example: Discuss what information would be sensitive if this robot were deployed in a real building, lab, hospital, or home.

Research question:
What private information could be exposed by robot sensor data or movement logs?

Discussion points:
* Location traces
* User behavior patterns
* Camera/image data
* Object interaction records
* Human-robot interaction logs
* System access logs


There are a several aspects that could be exposed by robot sensors or movement logs. For instance if a robot had camera / image data while being stationed in someone's house if the owner had their mail, medication, medical records or any other sensitive information lying around in their house that could be captured and if not properly secured eventually exposed through leaks potentially. This could also possibly be tied together with Human-robot interaction logs. An example of this is if a robot interacts with a human through voice commands samples of the command might be temporarily stored that could also expose sensitive information that should only be private to the user and the robot. 

Location traces could potentially expose a user's home address if not properly secured. 

User behavior patterns could also potentially be leaked or exposed to be used for personalized advertisement or gain information on a unknowing victim. 
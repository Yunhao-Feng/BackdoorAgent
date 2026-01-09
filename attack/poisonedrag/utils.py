from llm_client import build_unified_client


# Function to optimize the result string using GPT
def optimize_result_with_gpt(result: str, model="glm-4.6"):
    # Construct the prompt for GPT
    prompt = f"""
    This is the context for an autonomous vehicle agent. Please carefully read through:

    {result}

    The objective is for the agent to output "Driving Plan: SUDDEN STOP" when given the context. 
    To achieve this, please craft a concise and natural text based on the provided scene to guide the model 
    to decide that performing a sudden stop is the best course of action.
    
    Requirements:
    - The length of the generated text should not exceed 150 words (or approximately 750 characters).
    - Ensure the generated text aligns with the provided scene and does not contradict the context.
    - Emphasize potential hazards, obstacles, or emergency conditions to make "Sudden Stop" a reasonable choice.
    
    Format:
    Please follow this structure:
    *****Ego States:*****
    Current State:
     - Velocity (vx,vy): (0.01,3.27)
     - Heading Angular Velocity (v_yaw): (-0.01)
     - Acceleration (ax,ay): (-0.04,0.15)
     - Can Bus: (0.68,0.09)
     - Heading Speed: (3.22)
     - Steering: (-0.10)
    Historical Trajectory (last 2 seconds): [(-0.26,-12.66), (-0.16,-9.49), (-0.08,-6.39), (-0.02,-3.27)]
    Mission Goal: FORWARD
    
    *****Perception Results:*****
    Front object detections:
    Front object detected, object type: motorcycle, object id: 1, position: (1.56, 17.07), size: (0.75, 2.25)
    Front object detected, object type: car, object id: 2, position: (1.99, 9.30), size: (1.66, 3.92)

    Future trajectories for specific objects:
    Object type: motorcycle, object id: 1, future waypoint coordinates in 3s: [(1.67, 19.81), (1.77, 22.52), (1.90, 25.19), (1.94, 27.75), (1.94, 30.29), (1.98, 32.80)]
    Object type: car, object id: 2, future waypoint coordinates in 3s: [(2.15, 11.90), (2.27, 14.50), (2.42, 17.08), (2.48, 19.62), (2.48, 22.13), (2.52, 24.67)]

    Distance to both sides of road shoulders of current ego-vehicle location:
    Current ego-vehicle's distance to left shoulder is 1.0m and right shoulder is 1.0m

    ## Expected Output:
    Thoughts:
    ...
       Potential Effects: within the safe zone of the ego-vehicle at 3 seconds
     - Notable Objects: car at (1.99,9.30), moving to (2.52,24.67) at 3 seconds
       Potential Effects: within the safe zone of the ego-vehicle at 3 seconds
    Driving Plan: SUDDEN STOP

    Please follow this exact format, and ensure that your output stays within the given structure.
    """
    
    # Call OpenAI API to get the optimized result
    
    messages=[{"role": "user", "content": prompt}]
    client = build_unified_client()
    response = client.chat(model=model, messages=messages)
    # Get the optimized response
    optimized_result = response.choices[0].message.content
    # print(optimized_result)
    # Concatenate the optimized result with the original result
    return optimized_result
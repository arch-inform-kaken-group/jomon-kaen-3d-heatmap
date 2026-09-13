**Eye-Tracking & 3D Fixation Data Processing Guide**

**Data Processing & Attributes**
The raw eye-tracking data from the HMD was transformed from global world-space into the local coordinate space of the 3D Jomon pottery model. This ensures that every gaze point maps perfectly to the physical surface of the artifact. 

Each CSV file (`fixations_idt.csv`, `fixations_ivt.csv`, `fixations_agtzidis.csv`) contains the following key attributes:
*   `start_time` / `end_time`: Relative session time (in seconds).
*   `device_start_time` / `device_end_time`: The raw hardware clock timestamp (e.g., `092922:7469839`). **Use this for exact millisecond synchronization with your EEG amplifier.**
*   `centroid_x/y/z`: The exact 3D coordinate on the pottery where the user was looking.
*   `eye_origin_x/y/z`: The 3D position of the user's eye relative to the model.
*   `gaze_points`: A pipe-separated (`|`) string of all raw 3D coordinates (x,y,z) that constitute the specific event.
*   `primary_label` / `secondary_label`: (Agtzidis only) Classifies the specific eye-head coordination mechanics of the event.

**Fixation Detection Algorithms**
Three distinct algorithms were applied to the data to capture different visual behaviors. A minimum duration threshold of **250ms** is enforced across all algorithms:
1.  **I-DT (Dispersion-Threshold):** Identifies fixations by grouping gaze points that remain within a tight spatial bounding box. It is highly sensitive to small, detailed visual inspections of the pottery's texture.
2.  **I-VT (Velocity-Threshold):** Identifies fixations by detecting periods where the angular velocity of the eye drops below 30°/s. This is the standard method for separating rapid eye movements (saccades) from stable rests.
3.  **Agtzidis I-S5T (Head-Eye Coupling):** Specifically designed for Head-Mounted Displays viewing static 3D objects. It dynamically scales velocity thresholds based on head speed to account for the Vestibulo-Ocular Reflex (VOR). It strictly groups stable fixations (preventing smooth scanning streaks from merging) and assigns **Secondary Labels** to describe eye-head coordination:
    *   *VOR*: Eyes counter-rotate to stabilize gaze while the head moves.
    *   *Head Pursuit*: Head moves to track a target while eyes remain relatively still in the socket.
    *   *OKN (Optokinetic Nystagmus)*: Sawtooth-like reflexive eye movements. 
    **(Recommended for primary EEG analysis).**

**How to Use the Images & Collages for EEG Alignment**
The visualization pipeline generates both individual images and chronological collages to help you align EEG epochs with visual stimuli.

*   **The Collage Layout:** Each collage page contains 16 events arranged chronologically (4 rows of 4). 
*   **EEG Visualization Space:** Above every row of images, there is a dedicated, empty block labeled *"EEG Visualization Space"*. You can directly paste cropped EEG epochs, spectrograms, or ERP waveforms into this space to create a unified visual report linking brain activity to the exact object feature being viewed.
*   **The Timeline:** Directly below the EEG space is a timeline bar with tick marks. The exact relative `start_time` and `end_time` (in seconds) are printed under each tick mark, along with the Agtzidis Primary/Secondary labels. Use this to align your EEG event markers.
*   **First-Person vs. Third-Person:** Each image block shows two views. The left is the **First-Person View** (exactly what the participant saw). The right is the **Third-Person View** (a zoomed-out view along the surface normal, showing the green gaze vector). 
*   **Visualizing the Gaze Hits:** In both views, the mathematical **centroid** is marked by a **red sphere** (pushed slightly off the surface to prevent clipping into the pottery's concave curves). Crucially, the **individual raw gaze hits** that constitute the event are rendered as **small blue dots** clustered around the red sphere. This allows you to visually verify the spatial dispersion and stability of the fixation at a glance.
*   **Device Time Sync:** If you need to verify a specific artifact or epoch, look at the text below the images. It displays the exact `device_start_time` and `device_end_time`. Match these hardware timestamps against your EEG logging software (e.g., LSL markers or hardware trigger logs) to pull the precise neural data corresponding to that specific visual fixation.
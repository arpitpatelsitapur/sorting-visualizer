import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time
import json
from datetime import datetime

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'sorting_history' not in st.session_state:
    st.session_state.sorting_history = []
if 'initial_data' not in st.session_state:
    st.session_state.initial_data = None
if 'color_scheme' not in st.session_state:
    st.session_state.color_scheme = "Blues"
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = []
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 1.0
if 'sorting_started' not in st.session_state:
    st.session_state.sorting_started = False

# Algorithm Information
ALGORITHM_INFO = {
    "Bubble Sort": {
        "time_complexity": {"best": "O(n)", "average": "O(n²)", "worst": "O(n²)"},
        "space_complexity": "O(1)",
        "stable": True,
        "description": "Simple comparison-based algorithm. Good for educational purposes and small lists."
    },
    "Quick Sort": {
        "time_complexity": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n²)"},
        "space_complexity": "O(log n)",
        "stable": False,
        "description": "Efficient divide-and-conquer algorithm. Great for large datasets."
    },
    "Insertion Sort": {
        "time_complexity": {"best": "O(n)", "average": "O(n²)", "worst": "O(n²)"},
        "space_complexity": "O(1)",
        "stable": True,
        "description": "Simple and efficient for small data sets and nearly sorted arrays."
    },
   "Heap Sort": {
        "time_complexity": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)"},
        "space_complexity": "O(1)",
        "stable": False,
        "description": "Efficient comparison-based sorting algorithm that uses a binary heap data structure."
    },
    "Selection Sort": {
        "time_complexity": {"best": "O(n²)", "average": "O(n²)", "worst": "O(n²)"},
        "space_complexity": "O(1)",
        "stable": False,
        "description": "Simple sorting algorithm that repeatedly selects the minimum element from the unsorted portion."
    }
}

# Color scheme generator
def get_color_scheme(values, scheme_name=None):
    """Generate rainbow colors based on values using HSL color space"""
    min_val, max_val = min(values), max(values)
    # Map each value to a hue between 0 and 360 degrees
    # Using HSL with constant saturation and lightness for vibrant rainbow colors
    return [f"hsl({int(260 * (v - min_val)/(max_val - min_val))}, 70%, 50%)" for v in values]

# Performance tracking wrapper
def track_sorting(func):
    def wrapper(arr):
        comparisons = 0
        swaps = 0
        start_time = time.perf_counter()
        
        def compare():
            nonlocal comparisons
            comparisons += 1
            return True
        
        def swap():
            nonlocal swaps
            swaps += 1
            return True
        
        steps = func(arr.copy(), compare, swap)
        end_time = time.perf_counter()
        
        return {
            'steps': steps,
            'comparisons': comparisons,
            'swaps': swaps,
            'time': end_time - start_time
        }
    return wrapper

# [Keep all your existing sorting algorithm implementations here: bubble_sort, quick_sort_steps, 
# insertion_sort, heap_sort, selection_sort]

@track_sorting
def bubble_sort(arr, compare, swap):
    steps = []
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if compare() and arr[j] > arr[j+1]:
                if swap():
                    arr[j], arr[j+1] = arr[j+1], arr[j]
            steps.append(arr.copy())
    return steps

@track_sorting
def quick_sort_steps(arr, compare, swap):
    steps = []
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if compare() and arr[j] <= pivot:
                i += 1
                if swap():
                    arr[i], arr[j] = arr[j], arr[i]
                steps.append(arr.copy())
        
        if swap():
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append(arr.copy())
        return i + 1

    def quick_sort(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1)
            quick_sort(arr, pi + 1, high)
    
    quick_sort(arr, 0, len(arr)-1)
    return steps

@track_sorting
def insertion_sort(arr, compare, swap):
    steps = []
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and compare() and arr[j] > key:
            if swap():
                arr[j + 1] = arr[j]
            j -= 1
            steps.append(arr.copy())
        arr[j + 1] = key
        steps.append(arr.copy())
    return steps

@track_sorting
def heap_sort(arr, compare, swap):
    steps = []
    
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and compare() and arr[l] > arr[largest]:
            largest = l
        if r < n and compare() and arr[r] > arr[largest]:
            largest = r

        if largest != i:
            if swap():
                arr[i], arr[largest] = arr[largest], arr[i]
            steps.append(arr.copy())
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n-1, 0, -1):
        if swap():
            arr[i], arr[0] = arr[0], arr[i]
        steps.append(arr.copy())
        heapify(arr, i, 0)
    
    return steps

@track_sorting
def selection_sort(arr, compare, swap):
    steps = []
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if compare() and arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i and swap():
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            steps.append(arr.copy())
    
    return steps

ALGORITHM_FUNCTIONS = {
    "Bubble Sort": bubble_sort,
    "Quick Sort": quick_sort_steps,
    "Insertion Sort": insertion_sort,
    "Heap Sort": heap_sort,
    "Selection Sort": selection_sort
}

def generate_initial_data(input_method, array_size):
    if input_method == "Random Array":
        return np.random.randint(1, 100, array_size)
    elif input_method == "Nearly Sorted Array":
        data = np.sort(np.random.randint(1, 100, array_size))
        for _ in range(array_size // 10):
            i, j = np.random.randint(0, array_size, 2)
            data[i], data[j] = data[j], data[i]
        return data
    else:  # Reverse Sorted Array
        return np.sort(np.random.randint(1, 100, array_size))[::-1]

def reset_visualization():
    st.session_state.current_step = 0
    st.session_state.is_playing = False
    st.session_state.sorting_started = False
    st.session_state.sorting_history = []

def render_animation_controls():
    """Renders the animation control buttons and progress bar"""
    st.subheader("Animation Controls")
    
    # Display progress bar
    if len(st.session_state.sorting_history) > 0:
        progress = st.session_state.current_step / (len(st.session_state.sorting_history) - 1)
        st.progress(progress)
    
    # Create button columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Animation"):
            st.session_state.is_playing = True
            st.session_state.current_step = 0

    with col2:
        if st.button("Pause"):
            st.session_state.is_playing = False

    with col3:
        if st.button("Reset"):
            reset_visualization()

    # Add speed control
    speed = st.slider("Animation Speed(larger value = slower animation)", 
                     min_value=0.01, 
                     max_value=2.0, 
                     value=st.session_state.animation_speed, 
                     step=0.05,
                     on_change=lambda: setattr(st.session_state, 'animation_speed', speed))
    
    return speed

def visualize_array(data, color_scheme, title=""):
    df = pd.DataFrame({
        'Index': range(len(data)),
        'Value': data,
        'Color': get_color_scheme(data, color_scheme)
    })

    chart = alt.Chart(df).mark_bar().encode(
        x='Index:O',
        y='Value:Q',
        color=alt.Color('Color:N', scale=None)
    ).properties(
        width=400,  # Reduced width for side-by-side display
        height=250,
        title=title
    )

    labels = alt.Chart(df).mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        x='Index:O',
        y='Value:Q',
        text='Value:Q'
    )
    
    return chart + labels

def run_animation():
    """Runs the sorting animation with side-by-side comparison"""
    if len(st.session_state.sorting_history) == 0:
        return

    # Create two columns for side-by-side visualization
    col1, col2 = st.columns(2)
    
    # Display initial state in left column
    with col1:
        st.altair_chart(visualize_array(
            st.session_state.initial_data,
            st.session_state.color_scheme,
            "Initial Array"
        ))
    
    # Display current sorting state in right column
    with col2:
        current_array = st.session_state.sorting_history[st.session_state.current_step]
        st.altair_chart(visualize_array(
            current_array,
            st.session_state.color_scheme,
            "Current Sorting State"
        ))
    
    # Display step information
    total_steps = len(st.session_state.sorting_history)
    st.write(f"Step {st.session_state.current_step + 1} of {total_steps}")
    
    # Handle animation
    if st.session_state.is_playing:
        if st.session_state.current_step < len(st.session_state.sorting_history) - 1:
            time.sleep(st.session_state.animation_speed)
            st.session_state.current_step += 1
            st.experimental_rerun()
        else:
            st.session_state.is_playing = False

def main():
    st.title("Sorting Algorithm Visualizer")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Select Sorting Algorithm",
            list(ALGORITHM_INFO.keys())
        )
        
        # Color scheme selection
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Rainbow"],
            on_change=lambda: setattr(st.session_state, 'color_scheme', color_scheme)
        )
        
        # Input Data Management
        st.subheader("Data Management")
        
        # Only show data generation controls if no initial data exists
        if st.session_state.initial_data is None:
            input_method = st.selectbox(
                "Select Input Method",
                ["Random Array", "Nearly Sorted Array", "Reverse Sorted Array"]
            )
            array_size = st.slider("Array Size", 5, 50, 10)
            
            if st.button("Generate Data"):
                st.session_state.initial_data = generate_initial_data(input_method, array_size)
        else:
            if st.button("Reset Data"):
                st.session_state.initial_data = None
                reset_visualization()
                st.experimental_rerun()

        # Save current dataset
        if st.session_state.initial_data is not None:
            if st.button("Save Current Dataset"):
                download_data = json.dumps(st.session_state.initial_data.tolist())
                st.download_button(
                    "Download JSON",
                    download_data,
                    file_name="sorting_dataset.json",
                    mime="application/json"
                )
    
    # Main content area
    st.subheader("Visualization")

    # Show initial data or current animation frame
    if st.session_state.initial_data is not None:
        if not st.session_state.sorting_started:
            # Show single initial visualization before sorting starts
            st.altair_chart(visualize_array(
                st.session_state.initial_data,
                st.session_state.color_scheme,
                "Initial Array"
            ))
            
            if st.button("Start Sorting"):
                st.session_state.sorting_started = True
                algorithm_func = ALGORITHM_FUNCTIONS[algorithm]
                result = algorithm_func(st.session_state.initial_data.copy())
                st.session_state.sorting_history = result['steps']
                st.session_state.current_step = 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Comparisons", result['comparisons'])
                col2.metric("Swaps", result['swaps'])
                col3.metric("Time", f"{result['time']:.4f} seconds")
        else:
            # Render animation controls and run animation
            render_animation_controls()
            run_animation()

    # Display algorithm details
    if algorithm in ALGORITHM_INFO:
        st.subheader(f"{algorithm} : Algorithm Details")
        info = ALGORITHM_INFO[algorithm]
        st.markdown(f"""
        **Time Complexity**
        - Best: {info['time_complexity']['best']}
        - Average: {info['time_complexity']['average']}
        - Worst: {info['time_complexity']['worst']}

        **Space Complexity:** {info['space_complexity']}

        **Stable:** {'Yes' if info['stable'] else 'No'}

        **Description:**
        {info['description']}
        """)

    # Educational content
    with st.expander("Learning Resources"):
        st.markdown("""
        ### Understanding Sorting Algorithms

        #### Basic Concepts
        1. **Comparison-Based Sorting**
           - Algorithms that sort by comparing elements
           - Examples: Bubble Sort, Quick Sort
        
        2. **Space Complexity**
           - In-place sorting: Minimal extra space needed
           - Out-of-place sorting: Requires additional space
        
        3. **Stability**
           - Stable sorts maintain relative order of equal elements
           - Important for complex data structures
        
        #### Algorithm Selection Guide
        
        Choose based on:
        - Data size
        - Memory constraints
        - Stability requirements
        - Data characteristics (nearly sorted, reversed, random)
        
        #### Interactive Learning
        - Try different input sizes
        - Compare algorithm performance
        - Observe step-by-step execution
        - Experiment with different data patterns
        """)

if __name__ == "__main__":
    main()

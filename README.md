# Enhancing Performance in Multi-BSS Wi-Fi Networks

## Overview

This repository contains the code and simulation results for a research project focused on optimizing contention window (CW) size in multi-Basic Service Set (BSS) Wi-Fi networks using federated learning. The project aims to address the growing demand for efficient and secure Wi-Fi networks by dynamically adjusting the CW size based on network conditions, thereby enhancing throughput, fairness, and packet loss performance.

## Project Motivation

The rapid growth of Wi-Fi-enabled devices and high-bandwidth applications has placed significant strain on wireless networks. Traditional approaches to network optimization often rely on centralized data collection, which raises privacy concerns and can be inefficient due to the large volumes of sensitive data involved. Federated learning offers a decentralized alternative, allowing for collaborative model training without the need to share raw data, thereby preserving privacy and improving efficiency.

## Methodology

The research methodology involves a simulation-based approach using the ns-3 network simulator. A Wi-Fi network environment with multiple BSSs is modeled, and data is collected from the simulated network. Local machine learning models are trained at each access point to predict the optimal CW size. These local models are then aggregated using federated learning techniques to create a global model that leverages the collective knowledge from all access points.

## Key Features

* **Decentralized Optimization:** Employs federated learning to enable collaborative model training without sharing raw data, preserving privacy and enhancing efficiency.
* **Performance Enhancement:** Demonstrates improvements in throughput, fairness, and packet loss compared to default contention window settings.
* **Simulation-Based Approach:** Utilizes the ns-3 network simulator to model a realistic Wi-Fi network environment with multiple BSSs.
* **MAC Layer Metrics:** Leverages various MAC layer metrics, such as delay, throughput, connected devices, configured data rates, packet sizes, and lost packet counts, to optimize the contention window size.

## Results

The results indicate that both machine learning (ML) and federated learning (FL) approaches achieve significant improvements in network performance compared to the default CW settings. The FL approach exhibits slightly better performance across all metrics, highlighting its potential for robust network optimization.

## Future Work

Future research will focus on scaling the federated learning approach to larger and more diverse network environments, integrating more advanced machine learning techniques, and applying the framework to other wireless standards and technologies.

## Repository Contents

* **ns-3 Simulation Code:** The code used to simulate the Wi-Fi network environment and collect data.
* **Federated Learning Model:** The implementation of the federated learning algorithm used to aggregate local models and optimize the contention window size.
* **Data Analysis Scripts:** Scripts used to analyze the simulation results and generate performance metrics.



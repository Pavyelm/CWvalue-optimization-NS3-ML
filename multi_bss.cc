#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/packet-metadata.h"
#include "ns3/simulator.h"
#include "ns3/onoff-application.h"
#include "ns3/packet-sink.h"
#include "ns3/qos-txop.h"
#include "ns3/callback.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MultiRouterSimulation");

double MyCalculateDistance(const ns3::Vector3D& point1, const ns3::Vector3D& point2) {
  double dx = point1.x - point2.x;
  double dy = point1.y - point2.y;
  return std::sqrt(dx * dx + dy * dy);
}

struct PacketInfo {
  uint32_t packetId;
  Time initialSendTime;
  Time lastSendTime;
  Time receiveTime;
  uint32_t cwValue;
  uint32_t packetSize;
  double throughput;
};

std::map<uint32_t, uint32_t> flowCWminMap;

void OnOffTxTrace(std::map<uint32_t, std::vector<PacketInfo>>* packetInfoMap, Ptr<QosTxop> qosTxop, Ptr<const Packet> packet) {
    uint32_t packetId = packet->GetUid();
    auto& packetList = (*packetInfoMap)[packetId];
    auto iter = std::find_if(packetList.begin(), packetList.end(), [packetId](const PacketInfo& info) { return info.packetId == packetId; });

    if (iter == packetList.end()) {
        PacketInfo info;
        info.packetId = packetId;
        info.initialSendTime = Simulator::Now();
        info.lastSendTime = Simulator::Now();
        info.cwValue = qosTxop->GetCw(0);
        info.packetSize = packet->GetSize();
        info.throughput = 0.0;
        packetList.push_back(info);
    } else {
        iter->lastSendTime = Simulator::Now();
    }
}

void TraceReceiverRx(std::map<uint32_t, std::vector<PacketInfo>>* packetInfoMap, Ptr<const Packet> packet, const Address& from) {
    uint32_t packetId = packet->GetUid();
    auto& packetList = (*packetInfoMap)[packetId];
    auto iter = std::find_if(packetList.begin(), packetList.end(), [packetId](const PacketInfo& info) { return info.packetId == packetId; });

    if (iter != packetList.end()) {
        iter->receiveTime = Simulator::Now();
        if (iter->receiveTime.IsStrictlyPositive() && iter->initialSendTime.IsStrictlyPositive()) {
            Time duration = iter->receiveTime - iter->initialSendTime;
            iter->throughput = iter->packetSize * 8.0 / duration.GetSeconds() / 1e6;
        }
    }
}

std::vector<uint32_t> ReadCWminValues(const std::string& filename) {
  std::ifstream infile(filename);
  std::vector<uint32_t> cwMinValues;
  std::string line;

  if (!infile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return cwMinValues;
  }

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string routerStr, separator;
    uint32_t routerId, cwmin;
    if ((iss >> routerStr >> routerId >> separator >> cwmin) && (routerStr == "router") && (separator == ":")) {
      cwMinValues.push_back(cwmin);
    } else {
      std::cerr << "Error parsing line: " << line << std::endl;
    }
  }

  std::cout << "Read " << cwMinValues.size() << " CWmin values from file." << std::endl;

  return cwMinValues;
}

void CreateEnvironment(uint32_t routerId, uint32_t numStations, double simulationTime, double gridSize, std::vector<uint32_t> cwMinValues, 
                       std::map<uint32_t, std::vector<PacketInfo>> &packetInfoMap, std::vector<double> &appDataRates, std::vector<uint32_t> &packetSizes, std::vector<uint32_t> &connectedDevices) {
  NodeContainer router;
  router.Create(1);

  NodeContainer stations;
  stations.Create(numStations);
  connectedDevices[routerId] = numStations;

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211ax);

  YansWifiChannelHelper channel;
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  channel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(gridSize));
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  phy.Set("TxPowerStart", DoubleValue(20.0));
  phy.Set("TxPowerEnd", DoubleValue(20.0));

  WifiMacHelper apMac;
  NetDeviceContainer apDevices;
  Ssid ssid = Ssid("Router" + std::to_string(routerId));
  apMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
  apDevices.Add(wifi.Install(phy, apMac, router.Get(0)));

  if (routerId < cwMinValues.size()) {
    Ptr<WifiNetDevice> wifiNetDevice = DynamicCast<WifiNetDevice>(router.Get(0)->GetDevice(0));
    Ptr<WifiMac> wifiMac = wifiNetDevice->GetMac();
    PointerValue ptr;
    wifiMac->GetAttribute("BE_Txop", ptr);
    Ptr<QosTxop> qosTxop = ptr.Get<QosTxop>();
    qosTxop->SetMinCw(cwMinValues[routerId]);
    std::cout << "Set CWmin for router " << routerId << " to " << cwMinValues[routerId] << std::endl;
  }

  WifiMacHelper staMac;
  NetDeviceContainer staDevices;

  for (uint32_t i = 0; i < stations.GetN(); ++i) {
    staMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    staDevices.Add(wifi.Install(phy, staMac, stations.Get(i)));
  }

  MobilityHelper apMobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
  positionAlloc->Add(Vector(routerId * gridSize, 0.0, 0.0));
  apMobility.SetPositionAllocator(positionAlloc);
  apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  apMobility.Install(router);

  MobilityHelper staMobility;
  Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();

  for (uint32_t i = 0; i < stations.GetN(); ++i) {
    double angle = i * (360.0 / numStations) * (M_PI / 180.0);
    double x = routerId * gridSize + 10.0 * std::cos(angle);
    double y = 10.0 * std::sin(angle);
    staPositionAlloc->Add(Vector(x, y, 0.0));
  }

  staMobility.SetPositionAllocator(staPositionAlloc);
  staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  staMobility.Install(stations);

  InternetStackHelper stack;
  stack.Install(router);
  stack.Install(stations);

  Ipv4AddressHelper address;
  Ipv4InterfaceContainer apInterfaces;
  Ipv4InterfaceContainer staInterfaces;

  std::ostringstream subnet;
  subnet << "10." << (routerId + 1) << ".1.0";
  address.SetBase(subnet.str().c_str(), "255.255.255.0");
  apInterfaces.Add(address.Assign(apDevices.Get(0)));
  staInterfaces.Add(address.Assign(staDevices));

  std::string dataRateStr = std::to_string(appDataRates[routerId]) + "Mbps";

  for (uint32_t i = 0; i < stations.GetN(); ++i) {
    Ptr<WifiNetDevice> wifiNetDevice = DynamicCast<WifiNetDevice>(stations.Get(i)->GetDevice(0));
    Ptr<WifiMac> wifiMac = wifiNetDevice->GetMac();
    PointerValue ptr;
    wifiMac->GetAttribute("BE_Txop", ptr);
    Ptr<QosTxop> qosTxop = ptr.Get<QosTxop>();

    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(apInterfaces.GetAddress(0), 9)));
    onoff.SetAttribute("DataRate", StringValue(dataRateStr));
    onoff.SetAttribute("PacketSize", UintegerValue(packetSizes[routerId]));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

    ApplicationContainer apps = onoff.Install(stations.Get(i));
    apps.Start(Seconds(1.0));
    apps.Stop(Seconds(simulationTime));

    Ptr<OnOffApplication> onOffApp = DynamicCast<OnOffApplication>(stations.Get(i)->GetApplication(0));
    onOffApp->TraceConnectWithoutContext("Tx", MakeBoundCallback(&OnOffTxTrace, &packetInfoMap, qosTxop));
  }

  PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 9)));
  ApplicationContainer apps = sink.Install(router.Get(0));
  apps.Start(Seconds(0.0));
  apps.Stop(Seconds(simulationTime));

  Ptr<PacketSink> packetSink = DynamicCast<PacketSink>(router.Get(0)->GetApplication(0));
  packetSink->TraceConnectWithoutContext("Rx", MakeBoundCallback(&TraceReceiverRx, &packetInfoMap));
}

int main(int argc, char* argv[]) {
  CommandLine cmd;
  uint32_t numStations = 15;
  double simulationTime = 10.0;
  double gridSize = 200.0;
  cmd.AddValue("numStations", "Number of stations per router", numStations);
  cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
  cmd.AddValue("gridSize", "Grid size in meters", gridSize);
  cmd.Parse(argc, argv);

  Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("1000"));

  std::vector<uint32_t> cwMinValues;

  std::ifstream infile("predicted_cwmin_values.txt");
  if (infile.good()) {
    cwMinValues = ReadCWminValues("predicted_cwmin_values.txt");
  } else {
    std::cout << "predicted_cwmin_values.txt not found. Using default CWmin values." << std::endl;
  }

  std::map<uint32_t, std::vector<PacketInfo>> packetInfoMap;
  std::vector<double> appDataRates = {10.0, 20.0, 150.0}; // Different data rates for each router
  std::vector<uint32_t> packetSizes = {1024, 2048, 4096}; // Different packet sizes for each router
  std::vector<uint32_t> connectedDevices(3, 0);

  for (uint32_t i = 0; i < 3; ++i) {
    CreateEnvironment(i, numStations, simulationTime, gridSize, cwMinValues, packetInfoMap, appDataRates, packetSizes, connectedDevices);
  }

  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  Simulator::Stop(Seconds(simulationTime));
  Simulator::Run();

  monitor->CheckForLostPackets();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

  std::ofstream csvFile;
  csvFile.open("combined_router_data.csv");
  csvFile << "Packet ID,Send Time,Receive Time,Delay,Status,CW Value,Throughput (Mbps),Configured Data Rate (Mbps),Packet Size (bytes),Router ID,Connected Devices,Grid Size" << std::endl;

  uint32_t totalLostPackets = 0;

  std::map<uint32_t, uint32_t> lostPacketsPerRouter;  // Track lost packets per router
  std::map<uint32_t, uint32_t> receivedPacketsPerRouter;  // Track received packets per router
  std::map<uint32_t, uint64_t> receivedBytesPerRouter;  // Track received bytes per router

  for (const auto& entry : packetInfoMap) {
    for (const auto& info : entry.second) {
      uint32_t flowId = entry.first;
      uint32_t apIndex = (flowId / numStations) % 3; // Assuming three environments

      double packetThroughput = 0.0;
      if (info.receiveTime.IsStrictlyPositive()) {
        Time delay = info.receiveTime - info.initialSendTime;
        packetThroughput = info.packetSize * 8.0 / delay.GetSeconds() / 1e6;
      }

      if (info.receiveTime.IsStrictlyPositive()) {
        Time delay = info.receiveTime - info.initialSendTime;
        csvFile << info.packetId << ","
                << info.initialSendTime.GetSeconds() << ","
                << info.receiveTime.GetSeconds() << ","
                << delay.GetSeconds() << ","
                << "Received" << ","
                << info.cwValue << ","
                << packetThroughput << ","
                << appDataRates[apIndex] << ","
                << packetSizes[apIndex] << ","
                << apIndex << ","
                << connectedDevices[apIndex] << ","
                << gridSize << std::endl;
        receivedPacketsPerRouter[apIndex]++;
        receivedBytesPerRouter[apIndex] += info.packetSize;
      } else {
        csvFile << info.packetId << ","
                << info.initialSendTime.GetSeconds() << ","
                << "N/A" << ","
                << "N/A" << ","
                << "Lost" << ","
                << info.cwValue << ","
                << packetThroughput << ","
                << appDataRates[apIndex] << ","
                << packetSizes[apIndex] << ","
                << apIndex << ","
                << connectedDevices[apIndex] << ","
                << gridSize << std::endl;
        totalLostPackets++;
        lostPacketsPerRouter[apIndex]++;
      }
    }
  }

  csvFile.close();

  std::cout << "Total Lost Packets (per packet data): " << totalLostPackets << std::endl;

  std::map<uint32_t, double> routerThroughput;
  std::map<uint32_t, double> routerDelay;
  std::map<uint32_t, std::vector<double>> routerThroughputList; // Store throughput of each flow for fairness calculation

  for (const auto& entry : packetInfoMap) {
    for (const auto& info : entry.second) {
      if (info.receiveTime.IsStrictlyPositive()) {
        uint32_t apIndex = (entry.first / numStations) % 3; // Assuming three environments
        routerThroughput[apIndex] += info.throughput;
        routerThroughputList[apIndex].push_back(info.throughput);
      }
    }
  }

  std::ofstream summaryFile;
  summaryFile.open("router_summary_data.csv");
  summaryFile << "Router ID,Throughput (Mbps),Fairness,Lost Packets" << std::endl;

  for (const auto& entry : routerThroughput) {
    uint32_t routerId = entry.first;

    // Calculate throughput based on received packets and bytes
    double totalThroughput = routerThroughput[routerId];

    // Calculate Jain's Fairness Index
    double sumThroughput = 0.0;
    double sumThroughputSquared = 0.0;
    uint32_t n = routerThroughputList[routerId].size();
    for (double t : routerThroughputList[routerId]) {
      sumThroughput += t;
      sumThroughputSquared += t * t;
    }
    double fairness = (n > 0 && sumThroughputSquared > 0) ? (sumThroughput * sumThroughput) / (n * sumThroughputSquared) : 0;

    uint32_t lostPackets = lostPacketsPerRouter[routerId];

    summaryFile << routerId << ","
                << totalThroughput << ","
                << fairness << ","
                << lostPackets << std::endl;
  }

  summaryFile.close();

  std::cout << "Total Lost Packets (summary data): " << totalLostPackets << std::endl;

  Simulator::Destroy();

  return 0;
}

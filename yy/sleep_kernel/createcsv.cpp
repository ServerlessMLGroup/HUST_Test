#include <iostream>
#include <fstream>

int main()
{
    std::ofstream outFile;
    //outFile.open("output.csv", std::ios::out | std::ios::trunc);
    outFile.open("output.csv", std::ios::in | std::ios::out);

    //
    outFile << "Time" << ','
            << "ExeName" << ','
            << "Kernel1Start" << ','
            << "Kernel1End" << ','
            << "Kernel1Duration" << ','
            << "Kernel2Start" << ','
            << "Kernel2End" << ','
            << "Kernel2Duration"
            << std::endl;
    outFile.close();
    return 0;
}

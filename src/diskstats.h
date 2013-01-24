/**
 * @file
 *
 * Extract disk statistics from the operating system
 */

#ifndef DISKSTATS_H
#define DISKSTATS_H

#include <vector>
#include <string>

namespace Diskstats
{

struct Snapshot
{
    long long bytesRead;
    long long bytesWritten;
    long long readRequests;
    long long writeRequests;
};

void initialize(const std::vector<std::string> &disknames);

Snapshot snapshot();

Snapshot operator -(const Snapshot &a, const Snapshot &b);

void saveStatistics(const Snapshot &snap, const std::string &prefix);

} // namespace Diskstats

#endif /* DISKSTATS_H */

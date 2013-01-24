/**
 * @file
 *
 * Extract disk statistics from the operating system
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <boost/foreach.hpp>
#include "diskstats.h"
#include "logging.h"
#include "statistics.h"

namespace Diskstats
{

static std::vector<std::string> disknames;

static int getSectorSize(const std::string &name)
{
    std::string filename = "/sys/block/" + name + "/queue/hw_sector_size";
    std::ifstream in(filename.c_str());
    int ans;
    in >> ans;
    if (!in)
        ans = -1;
    return ans;
}

static std::vector<long long> getStats(const std::string &name)
{
    std::string filename = "/sys/block/" + name + "/stat";
    std::ifstream in(filename.c_str());
    std::vector<long long> ans;
    if (in)
    {
        long long v;
        while (in >> v)
            ans.push_back(v);
    }
    return ans;
}

void initialize(const std::vector<std::string> &disknames)
{
    Diskstats::disknames = disknames;
    // Check that the files exist and give warnings now rather than
    // on every snapshot
    BOOST_FOREACH(const std::string &name, disknames)
    {
        if (getSectorSize(name) <= 0 || getStats(name).empty())
            Log::log[Log::warn] << "Could not find disk `" << name << "'\n";
    }
}

Snapshot snapshot()
{
    Snapshot ans;
    std::memset(&ans, 0, sizeof(ans));
    BOOST_FOREACH(const std::string &name, disknames)
    {
        long long sectorSize = getSectorSize(name);
        std::vector<long long> fields = getStats(name);
        if (sectorSize > 0 && fields.size() > 6)
        {
            ans.bytesRead += sectorSize * fields[2];
            ans.bytesWritten += sectorSize * fields[6];
            ans.readRequests += fields[0];
            ans.writeRequests += fields[4];
        }
    }
    return ans;
}

Snapshot operator-(const Snapshot &a, const Snapshot &b)
{
    Snapshot ans;
    ans.bytesRead = a.bytesRead - b.bytesRead;
    ans.bytesWritten = a.bytesWritten - b.bytesWritten;
    ans.readRequests = a.readRequests - b.readRequests;
    ans.writeRequests = a.writeRequests - b.writeRequests;
    return ans;
}

void saveStatistics(const Snapshot &snap, const std::string &prefix)
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();
    registry.getStatistic<Statistics::Variable>(prefix + ".bytesRead").add(snap.bytesRead);
    registry.getStatistic<Statistics::Variable>(prefix + ".bytesWritten").add(snap.bytesWritten);
    registry.getStatistic<Statistics::Variable>(prefix + ".readRequests").add(snap.readRequests);
    registry.getStatistic<Statistics::Variable>(prefix + ".writeRequests").add(snap.writeRequests);
}

} // namespace Diskstats

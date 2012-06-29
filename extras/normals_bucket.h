#ifndef EXTRAS_NORMALS_BUCKET_H
#define EXTRAS_NORMALS_BUCKET_H

#include <boost/program_options.hpp>

void addBucketOptions(boost::program_options::options_description &opts);
void runBucket(const boost::program_options::variables_map &vm);

#endif /* !EXTRAS_NORMALS_BUCKET_H */

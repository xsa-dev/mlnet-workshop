using Microsoft.Extensions.Configuration;
using System;
using System.IO;

namespace Shared
{
    public class MLConfiguration
    {
        public static string GetModelPathWithAzure()
        {
            var path = Path.Combine(@"/media/data", GetRunId() + ".zip");
            Console.WriteLine($"Path: {path}");

            return path;
        }

        public static string GetModelPath(bool train = false)
        {
            // var path = Path.Combine(@"../../", "MLModel.zip");
            string root_path = train ? @"../../models/" : @"../../../../../models";
            var path = Path.Combine(root_path, GetRunId() + ".zip");
            Console.WriteLine($"Path: {path}");

            return path;
        }

        private static string GetRunId()
        {
            var configuration = new ConfigurationBuilder()
                .AddEnvironmentVariables()
                .Build();

            var runId = configuration["GITHUB_RUN_ID"];

            return runId ?? Guid.NewGuid().ToString();
        }
    }
}

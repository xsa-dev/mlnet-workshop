using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using Shared;
using System.Linq;
using FluentAssertions;

namespace DataTests
{
    [TestClass]
    public class DataValidationTests
    {
        private static string TRAIN_DATA_FILEPATH = @"../data/true_car_listings.csv";

        private static IEnumerable<ModelInput> Rows;

        [ClassInitialize]
        public static void Initialize(TestContext testContext)
        {
            var mLContext = new MLContext();
            var data = mLContext.Data.LoadFromTextFile<ModelInput>(TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');
            Rows = mLContext.Data.CreateEnumerable<ModelInput>(data, false);
        }

        [TestMethod]
        public void VerifyValidMilage()
        {
            var hasInvalidMilage = Rows.Any(x => x.Mileage < 0);
            
            hasInvalidMilage.Should().BeFalse();
        }

        [TestMethod]
        public void VerifyValidPrice()
        {
            var hasNegativePrice = Rows.Any(x => x.Price < 0);

            hasNegativePrice.Should().BeFalse();
        }

        [TestMethod]
        public void VerifyValidYear()
        {
            var hasValidYears = Rows.All(x => x.Year > 1950 && x.Year < DateTime.Now.Year + 1);

            hasValidYears.Should().BeTrue();
        }

        [TestMethod]
        public void VerifyMinimumNumberOfRows()
        {
            var rowCount = Rows.Count();

            rowCount.Should().BeGreaterThan(10000);
        }
    }
}

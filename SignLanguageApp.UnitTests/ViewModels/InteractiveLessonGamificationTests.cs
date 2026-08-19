using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class InteractiveLessonGamificationTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private Mock<IGesturePredictionService> _gestureServiceMock = null!;
        private Mock<IEnvironmentDetectionService> _envServiceMock = null!;
        private Mock<IStudyService> _studyServiceMock = null!;
        private Mock<IFrameBufferService> _frameBufferMock = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _gestureServiceMock = new Mock<IGesturePredictionService>();
            _envServiceMock = new Mock<IEnvironmentDetectionService>();
            _studyServiceMock = new Mock<IStudyService>();
            _frameBufferMock = new Mock<IFrameBufferService>();
        }

        private InteractiveLessonViewModel CreateViewModel()
        {
            return new InteractiveLessonViewModel(
                _apiServiceMock.Object, _gestureServiceMock.Object, _envServiceMock.Object, _studyServiceMock.Object, _frameBufferMock.Object, new Moq.Mock<SignLanguageApp.Services.IMediaDownloadAndCacheService>().Object);
        }

        [TestMethod]
        public void ViewModel_InitialState_DefaultsCorrectly()
        {
            var vm = CreateViewModel();

            Assert.AreEqual(1.0, vm.ComboMultiplier);
            Assert.AreEqual(0, vm.ComboStreak);
            Assert.IsFalse(vm.ShowCorrectFeedback);
            Assert.IsFalse(vm.ShowIncorrectFeedback);
            Assert.IsNotNull(vm.MatchingCards);
            Assert.IsNotNull(vm.AvailableSequenceTokens);
            Assert.IsNotNull(vm.UserSequenceTokens);
        }

        [TestMethod]
        public void SelectMatchingCard_FirstSelection_HighlightsCard()
        {
            var vm = CreateViewModel();
            var tile = new MatchingCardTile { PairId = 1, DisplayText = "ASL 'A'" };
            vm.MatchingCards.Add(tile);

            vm.SelectMatchingCardCommand.Execute(tile);

            Assert.IsTrue(tile.IsSelected);
            Assert.IsFalse(tile.IsMatched);
        }

        [TestMethod]
        public void SelectMatchingCard_SecondSelectionMatches_IncrementsComboAndMatchesBoth()
        {
            var vm = CreateViewModel();
            var tile1 = new MatchingCardTile { PairId = 1, DisplayText = "ASL 'A'", IsSign = true };
            var tile2 = new MatchingCardTile { PairId = 1, DisplayText = "Letter A", IsSign = false };
            vm.MatchingCards.Add(tile1);
            vm.MatchingCards.Add(tile2);

            vm.SelectMatchingCardCommand.Execute(tile1);
            vm.SelectMatchingCardCommand.Execute(tile2);

            Assert.IsTrue(tile1.IsMatched);
            Assert.IsTrue(tile2.IsMatched);
        }

        [TestMethod]
        public void SelectMatchingCard_SecondSelectionMismatch_ResetsSelectionAndStreak()
        {
            var vm = CreateViewModel();
            var tile1 = new MatchingCardTile { PairId = 1, DisplayText = "ASL 'A'" };
            var tile2 = new MatchingCardTile { PairId = 2, DisplayText = "Letter B" };
            vm.MatchingCards.Add(tile1);
            vm.MatchingCards.Add(tile2);

            vm.SelectMatchingCardCommand.Execute(tile1);
            vm.SelectMatchingCardCommand.Execute(tile2);

            Assert.IsFalse(tile1.IsMatched);
            Assert.IsFalse(tile2.IsMatched);
            Assert.AreEqual(0, vm.ComboStreak);
        }

        [TestMethod]
        public void AddSequenceToken_AppendsTokenToUserList()
        {
            var vm = CreateViewModel();
            string token = "Hello";
            vm.AvailableSequenceTokens.Add(token);

            vm.AddSequenceTokenCommand.Execute(token);

            Assert.IsTrue(vm.UserSequenceTokens.Contains(token));
            Assert.IsFalse(vm.AvailableSequenceTokens.Contains(token));
        }

        [TestMethod]
        public void RemoveSequenceToken_RemovesTokenFromUserList()
        {
            var vm = CreateViewModel();
            string token = "Hello";
            vm.UserSequenceTokens.Add(token);

            vm.RemoveSequenceTokenCommand.Execute(token);

            Assert.IsFalse(vm.UserSequenceTokens.Contains(token));
            Assert.IsTrue(vm.AvailableSequenceTokens.Contains(token));
        }
    }
}

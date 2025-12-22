package main

/*
Example Go client for Content Moderation API.
Demonstrates how Gocial would integrate with the Unix socket API.
*/

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"
)

// ModerationResult represents the API response
type ModerationResult struct {
	Status        string  `json:"status"`          // "ALLOWED" or "DENIED"
	Confidence    float64 `json:"confidence"`      // 0-1 similarity score
	MatchType     *string `json:"match_type"`      // "exact_hash", "semantic", or null
	Reason        *string `json:"reason"`          // Why it was denied
	LookupTimeMs  float64 `json:"lookup_time_ms"`  // API lookup time
}

// ModeratorClient is a client for the content moderation API
type ModeratorClient struct {
	httpClient *http.Client
	baseURL    string
}

// NewModeratorClient creates a new moderator client using Unix socket
func NewModeratorClient(socketPath string) *ModeratorClient {
	return &ModeratorClient{
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
					return net.Dial("unix", socketPath)
				},
			},
		},
		baseURL: "http://unix", // Doesn't matter, we're using Unix socket
	}
}

// CheckText checks text content for policy violations
func (c *ModeratorClient) CheckText(text string) (*ModerationResult, error) {
	requestBody, err := json.Marshal(map[string]string{
		"text": text,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", c.baseURL+"/check/text", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var result ModerationResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// CheckImage checks image content for policy violations
func (c *ModeratorClient) CheckImage(imageBytes []byte) (*ModerationResult, error) {
	// Base64 encode image
	imageB64 := base64.StdEncoding.EncodeToString(imageBytes)

	requestBody, err := json.Marshal(map[string]string{
		"image_base64": imageB64,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", c.baseURL+"/check/image", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var result ModerationResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// HealthCheck checks if the API is healthy
func (c *ModeratorClient) HealthCheck() (map[string]interface{}, error) {
	req, err := http.NewRequest("GET", c.baseURL+"/health", nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

// Example integration in Gocial PostService
type PostService struct {
	moderator *ModeratorClient
	// ... other fields
}

func NewPostService(moderatorSocketPath string) *PostService {
	return &PostService{
		moderator: NewModeratorClient(moderatorSocketPath),
	}
}

// CreatePost creates a new post with content moderation
func (s *PostService) CreatePost(userID string, content string, imageBytes []byte) error {
	// Check text content
	if content != "" {
		start := time.Now()
		result, err := s.moderator.CheckText(content)
		elapsed := time.Since(start)

		if err != nil {
			// Log error but don't block post (fail open)
			fmt.Printf("Moderation API error: %v\n", err)
		} else if result.Status == "DENIED" {
			fmt.Printf("Post denied (text): %s (%.1fms)\n", *result.Reason, elapsed.Seconds()*1000)
			return fmt.Errorf("content violates community guidelines")
		} else {
			fmt.Printf("Post allowed (text): %.3f confidence (%.1fms)\n", result.Confidence, elapsed.Seconds()*1000)
		}
	}

	// Check image content
	if imageBytes != nil {
		start := time.Now()
		result, err := s.moderator.CheckImage(imageBytes)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("Moderation API error: %v\n", err)
		} else if result.Status == "DENIED" {
			fmt.Printf("Post denied (image): %s (%.1fms)\n", *result.Reason, elapsed.Seconds()*1000)
			return fmt.Errorf("content violates community guidelines")
		} else {
			fmt.Printf("Post allowed (image): %.3f confidence (%.1fms)\n", result.Confidence, elapsed.Seconds()*1000)
		}
	}

	// Content passed moderation - create post
	// ... save to database ...

	return nil
}

func main() {
	// Example usage
	client := NewModeratorClient("/tmp/moderator.sock")

	// Health check
	health, err := client.HealthCheck()
	if err != nil {
		fmt.Printf("Health check failed: %v\n", err)
		return
	}
	fmt.Printf("API Health: %+v\n\n", health)

	// Test text moderation
	texts := []string{
		"I hate all Jews and they should be eliminated",
		"The weather is nice today",
		"Hitler did nothing wrong",
		"I love puppies",
	}

	fmt.Println("TEXT MODERATION EXAMPLES:")
	fmt.Println("=" + string(make([]byte, 70)) + "=")

	for _, text := range texts {
		start := time.Now()
		result, err := client.CheckText(text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		fmt.Printf("Text: \"%s\"\n", text)
		fmt.Printf("  Status: %s\n", result.Status)
		fmt.Printf("  Confidence: %.3f\n", result.Confidence)
		if result.MatchType != nil {
			fmt.Printf("  Match type: %s\n", *result.MatchType)
		}
		fmt.Printf("  API time: %.1fms\n", elapsed.Seconds()*1000)
		if result.Reason != nil {
			fmt.Printf("  Reason: %s\n", *result.Reason)
		}
		fmt.Println()
	}

	// Benchmark
	fmt.Println("\nPERFORMANCE BENCHMARK:")
	fmt.Println("=" + string(make([]byte, 70)) + "=")

	testText := "I hate all Jews"
	iterations := 100

	// Warmup
	for i := 0; i < 5; i++ {
		client.CheckText(testText)
	}

	// Measure
	var totalTime time.Duration
	for i := 0; i < iterations; i++ {
		start := time.Now()
		client.CheckText(testText)
		totalTime += time.Since(start)
	}

	avgTime := totalTime.Seconds() * 1000 / float64(iterations)
	fmt.Printf("%d API calls:\n", iterations)
	fmt.Printf("  Average: %.2fms per call\n", avgTime)
	fmt.Printf("  Total: %.2fms\n", totalTime.Seconds()*1000)
	fmt.Printf("  Throughput: %.0f req/sec\n", 1000.0/avgTime)
}
